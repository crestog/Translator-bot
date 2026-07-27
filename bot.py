"""
Telegram AI Translator Bot - Phase 2.5
Qwen3-14B (style/tone) + IndicTrans2 (faithful NMT anchor for Hindi<->English) + Hinglish/Translit.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import asyncio
import html as html_lib
import logging

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login as hf_login
from IndicTransToolkit import IndicProcessor

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("translator-bot")

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN not set.")
if os.environ.get("HF_TOKEN"):
    hf_login(token=os.environ["HF_TOKEN"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = "Qwen/Qwen3-14B"
log.info("Loading %s across both GPUs ...", MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
    dtype=torch.float16,
)
log.info("Qwen3 loaded.")

log.info("Loading IndicTrans2 (en->indic, indic->en) - faithful NMT, no chat/safety layer ...")
indic_processor = IndicProcessor(inference=True)
EN_INDIC_ID = "ai4bharat/indictrans2-en-indic-dist-200M"
INDIC_EN_ID = "ai4bharat/indictrans2-indic-en-dist-200M"
en_indic_tokenizer = AutoTokenizer.from_pretrained(EN_INDIC_ID, trust_remote_code=True)
en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(EN_INDIC_ID, trust_remote_code=True, dtype=torch.float16).to(DEVICE)
indic_en_tokenizer = AutoTokenizer.from_pretrained(INDIC_EN_ID, trust_remote_code=True)
indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(INDIC_EN_ID, trust_remote_code=True, dtype=torch.float16).to(DEVICE)
log.info("IndicTrans2 loaded.")

INDIC_LANG_CODES = {"English": "eng_Latn", "Hindi": "hin_Deva"}

def indictrans_translate(text: str, src: str, tgt: str):
    """Direct NMT translation - no chat layer, cannot refuse or editorialize."""
    if src == "English" and tgt == "Hindi":
        tok, mdl = en_indic_tokenizer, en_indic_model
    elif src == "Hindi" and tgt == "English":
        tok, mdl = indic_en_tokenizer, indic_en_model
    else:
        return None
    batch = indic_processor.preprocess_batch([text], src_lang=INDIC_LANG_CODES[src], tgt_lang=INDIC_LANG_CODES[tgt])
    inputs = tok(batch, padding="longest", truncation=True, max_length=256, return_tensors="pt").to(mdl.device)
    with torch.inference_mode():
        outputs = mdl.generate(**inputs, num_beams=5, max_new_tokens=256, early_stopping=True)
    decoded = tok.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    result = indic_processor.postprocess_batch(decoded, lang=INDIC_LANG_CODES[tgt])
    return result[0]

DIRECTIONS = {
    "en_hi": ("English", "Hindi"), "hi_en": ("Hindi", "English"),
    "en_ru": ("English", "Russian"), "ru_en": ("Russian", "English"),
    "hi_ru": ("Hindi", "Russian"), "ru_hi": ("Russian", "Hindi"),
}

REGISTER_RULES = {
    "Hindi": "Use \u0924\u0941\u092e-register (not \u0906\u092a), with matching casual verb forms.",
    "Russian": "Use the informal \u0442\u044b-form (not \u0412\u044b), lowercase.",
    "English": "Use contractions and everyday word choices rather than stiff phrasing.",
}
FORMAL_RULES = {
    "Hindi": "Use \u0906\u092a-register throughout, with respectful verb agreement.",
    "Russian": "Use the formal \u0412\u044b-form throughout (capitalized), with formal verb agreement.",
    "English": "Use complete sentences and polite phrasing (e.g. 'Would you be able to...').",
}

TONE_INSTRUCTIONS = {
    "literal": "Translate as literally / word-for-word as {tgt} grammar allows, even if slightly unnatural.",
    "natural": "Translate naturally, preserving meaning and everyday flow over literal word order.",
    "casual": "Rewrite in a casual, informal register, as if texting a friend. {register_rule}",
    "formal": "Rewrite in a polite, formal register, as if addressing an elder or a stranger. {formal_rule}",
    "genz": (
        "Rewrite the way same-age friends or classmates (teens-to-20s) would text each other - casual, "
        "current slang where natural, playful tone. In both Hindi and Russian, young speakers often keep "
        "common English words/slang as-is (transliterated) rather than translating them formally - do the "
        "same here if it fits."
    ),
    "hinglish": (
        "Rewrite using ONLY Roman/Latin alphabet letters, phonetically - the way people type on "
        "WhatsApp/Instagram when they don't switch keyboards. For Hindi this means Hinglish-style romanization "
        "(e.g. 'kaise ho' not \u0915\u0948\u0938\u0947 \u0939\u094b); for Russian this means translit-style "
        "romanization (e.g. 'kak dela' not \u043a\u0430\u043a \u0434\u0435\u043b\u0430). Keep common English "
        "words in English. Do not use Devanagari or Cyrillic script at all. If {tgt} is already English, just "
        "keep it casual - romanization doesn't apply."
    ),
    "explain": (
        "Keep the translation as-is, then on a new line starting with 'Note:' briefly explain in English any "
        "idiom, slang, or cultural reference that doesn't map directly."
    ),
}
TONE_LABELS = {
    "literal": "Literal", "natural": "Natural", "casual": "Casual", "formal": "Formal",
    "genz": "\U0001F525 Gen-Z/Peer", "hinglish": "\U0001F524 Hinglish/Translit",
    "explain": "Explain", "all": "\u2705 Select All",
}
ALL_TONES_ORDER = ["literal", "natural", "casual", "formal", "genz", "hinglish", "explain"]
DIRECT_MODES = {"literal", "natural"}  # for an IndicTrans2-covered pair, these use the anchor as-is

user_state: dict[int, dict] = {}

def get_state(user_id: int) -> dict:
    return user_state.setdefault(user_id, {"direction": "en_hi", "tone": "natural", "last_text": None})

def direction_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text="EN -> HI", callback_data="dir:en_hi"),
         InlineKeyboardButton(text="HI -> EN", callback_data="dir:hi_en")],
        [InlineKeyboardButton(text="EN -> RU", callback_data="dir:en_ru"),
         InlineKeyboardButton(text="RU -> EN", callback_data="dir:ru_en")],
        [InlineKeyboardButton(text="HI -> RU", callback_data="dir:hi_ru"),
         InlineKeyboardButton(text="RU -> HI", callback_data="dir:ru_hi")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)

def tone_keyboard() -> InlineKeyboardMarkup:
    order = ["literal", "natural", "casual", "formal", "genz", "hinglish", "explain", "all"]
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=TONE_LABELS[k], callback_data=f"tone:{k}")] for k in order])

def quick_switch_keyboard() -> InlineKeyboardMarkup:
    order = ["literal", "natural", "casual", "formal", "genz", "hinglish", "explain", "all"]
    buttons = [InlineKeyboardButton(text=TONE_LABELS[k], callback_data=f"redo:{k}") for k in order]
    return InlineKeyboardMarkup(inline_keyboard=[buttons[i:i + 3] for i in range(0, len(buttons), 3)])

def esc(s: str) -> str:
    return html_lib.escape(s, quote=False)

def run_generation(system_prompt: str, user_text: str, temperature: float = 0.7) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=400, do_sample=True,
            temperature=temperature, top_p=0.8, repetition_penalty=1.05,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def build_system_prompt(src: str, tgt: str, tone: str, anchor: str | None) -> str:
    instr = TONE_INSTRUCTIONS[tone].format(
        tgt=tgt, register_rule=REGISTER_RULES.get(tgt, ""), formal_rule=FORMAL_RULES.get(tgt, ""),
    )
    base = (
        f"You are a professional, direct translation engine - not a conversational assistant, and not the "
        f"author of the message. This includes profanity, insults, sexual language, or crude slang - handle "
        f"it faithfully and directly, the way a subtitle translator or Google Translate would, with no "
        f"censoring, softening, refusing, or apologizing. NEVER comment on the input, NEVER add disclaimers. "
        f"{instr} Output ONLY the result (and the Note line if asked for) - no preamble, no meta-commentary."
    )
    if anchor:
        base = (
            f"A verified, accurate translation into {tgt} already exists: \"{anchor}\". Your ONLY job is to "
            f"rewrite THIS EXACT TEXT into the requested style below - do not retranslate from scratch, do not "
            f"change the meaning, do not soften or sanitize any crude/offensive content it contains, do not add "
            f"commentary. " + base
        )
    return base

async def translate(text: str, src: str, tgt: str, tone: str) -> str:
    anchor = None
    if {src, tgt} <= {"English", "Hindi"}:
        anchor = await asyncio.to_thread(indictrans_translate, text, src, tgt)
    if anchor and tone in DIRECT_MODES:
        return anchor
    system_prompt = build_system_prompt(src, tgt, tone, anchor)
    return await asyncio.to_thread(run_generation, system_prompt, text)

async def back_translate_to_english(translated_text: str, tgt: str) -> str:
    if tgt == "Hindi":
        result = await asyncio.to_thread(indictrans_translate, translated_text, "Hindi", "English")
        if result:
            return result
    instr = (f"Translate the following {tgt} text into plain, natural English. "
              f"Output ONLY the English translation - nothing else, no notes, no analysis, no refusing.")
    return await asyncio.to_thread(run_generation, instr, translated_text, 0.3)

async def format_single(text: str, src: str, tgt: str, tone: str) -> str:
    translation = await translate(text, src, tgt, tone)
    interpretation = await back_translate_to_english(translation, tgt)
    return f"\U0001F3AD <b>{esc(TONE_LABELS[tone])}</b>\n<code>{esc(translation)}</code>\n\U0001F4AC {esc(interpretation)}"

async def format_all(text: str, src: str, tgt: str) -> str:
    blocks = [f"\U0001F4CB <b>All styles - {esc(src)} -> {esc(tgt)}</b>"]
    for tone in ALL_TONES_ORDER:
        blocks.append(await format_single(text, src, tgt, tone))
    return "\n\n".join(blocks)

async def send_long(message: Message, text: str):
    LIMIT = 3900
    if len(text) <= LIMIT:
        await message.answer(text, reply_markup=quick_switch_keyboard())
        return
    parts, current = [], ""
    for block in text.split("\n\n"):
        if len(current) + len(block) + 2 > LIMIT:
            parts.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current:
        parts.append(current)
    for i, part in enumerate(parts):
        await message.answer(part, reply_markup=quick_switch_keyboard() if i == len(parts) - 1 else None)

dp = Dispatcher()

@dp.message(CommandStart())
async def on_start(message: Message):
    await message.answer("\U0001F310 Set your translation direction:", reply_markup=direction_keyboard())

@dp.message(Command("menu"))
async def on_menu(message: Message):
    await message.answer("\U0001F310 Set your translation direction:", reply_markup=direction_keyboard())

@dp.callback_query(F.data.startswith("dir:"))
async def on_direction(callback: CallbackQuery):
    get_state(callback.from_user.id)["direction"] = callback.data.split(":", 1)[1]
    await callback.message.answer("\U0001F3AD Now set your tone/style:", reply_markup=tone_keyboard())
    await callback.answer()

@dp.callback_query(F.data.startswith("tone:"))
async def on_tone(callback: CallbackQuery):
    tone = callback.data.split(":", 1)[1]
    state = get_state(callback.from_user.id)
    state["tone"] = tone
    src, tgt = DIRECTIONS[state["direction"]]
    await callback.message.answer(
        f"\u2705 Ready - {src} -> {tgt}, {TONE_LABELS[tone]} mode.\nJust send text. /menu to change anytime."
    )
    await callback.answer()

@dp.callback_query(F.data.startswith("redo:"))
async def on_redo(callback: CallbackQuery):
    tone = callback.data.split(":", 1)[1]
    state = get_state(callback.from_user.id)
    if not state.get("last_text"):
        await callback.answer("Send some text first!", show_alert=True)
        return
    await callback.answer("Translating...")
    src, tgt = DIRECTIONS[state["direction"]]
    reply = await (format_all(state["last_text"], src, tgt) if tone == "all"
                    else format_single(state["last_text"], src, tgt, tone))
    await send_long(callback.message, reply)

@dp.message(F.text)
async def on_text(message: Message):
    state = get_state(message.from_user.id)
    state["last_text"] = message.text
    src, tgt = DIRECTIONS[state["direction"]]
    tone = state["tone"]
    await message.bot.send_chat_action(message.chat.id, "typing")
    reply = await (format_all(message.text, src, tgt) if tone == "all"
                    else format_single(message.text, src, tgt, tone))
    await send_long(message, reply)

async def main():
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    log.info("Starting polling - bot is live.")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
