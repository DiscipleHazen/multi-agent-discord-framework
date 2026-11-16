import os
import sys
import time
import json
import types
import asyncio
import tempfile
import unittest
from io import BytesIO
from PIL import Image, ImageDraw

# Provide a lightweight mock for `discord` when running unit tests in headless CI environments
if "discord" not in sys.modules:
    mock_discord = types.ModuleType("discord")
    mock_discord.Message = object
    mock_discord.Member = object
    mock_discord.RawReactionActionEvent = object
    mock_discord.Client = object
    mock_discord.File = object
    mock_discord.Interaction = object

    class _MockEmbed:
        def __init__(self, title="", description="", color=0):
            self.title = title
            self.description = description
            self.color = color
            self.fields = []

        def add_field(self, name="", value="", inline=False):
            self.fields.append({"name": name, "value": value, "inline": inline})

    mock_discord.Embed = _MockEmbed
    mock_discord.Intents = type(
        "Intents",
        (),
        {"default": staticmethod(lambda: types.SimpleNamespace(message_content=True, members=True, reactions=True))},
    )
    sys.modules["discord"] = mock_discord

import config
import core_system
import text_and_parsers
import subconscious_engine
import memory_and_context
import media_engine
import ai_services
import bot_discovery
import commands_and_embeds
import main_bot
import orchestrator


class TestCoreAndParsers(unittest.TestCase):
    """Tests Step 1.1 (`core_system.py` EWMA Latency & `text_and_parsers.py` [THOUGHT:], Drift Guard & AST Tools)."""

    def test_ewma_latency_tracker_and_failover_ranking(self):
        core_system.record_model_outcome("completions:ModelSlow", success=True, elapsed_sec=6.5, prompt_tokens=120, completion_tokens=60)
        core_system.record_model_outcome("completions:ModelFast", success=True, elapsed_sec=0.8, prompt_tokens=100, completion_tokens=50)
        ranked = core_system.rank_models_by_ewma_latency(["ModelSlow", "ModelFast"], endpoint_label="completions")
        self.assertEqual(ranked[0], "ModelFast")
        tel = core_system.get_all_model_telemetry()
        self.assertIn("completions:ModelFast", tel)
        self.assertEqual(tel["completions:ModelFast"]["circuit_status"], "CLOSED")

    def test_thought_tag_extraction_and_media_rescue(self):
        raw = "[THOUGHT: He is testing my patience, so I will tease him dryly. [IMAGE_GEN: a smug wizard]] *smirks* Clearly you need practice."
        thought, cleaned = text_and_parsers.extract_and_strip_thought_tag(raw)
        self.assertIn("testing my patience", thought)
        self.assertNotIn("[THOUGHT:", cleaned)
        self.assertIn("[IMAGE_GEN: a smug wizard]", cleaned)

    def test_persona_signature_drift_guard(self):
        drifted = "Certainly! Here's a breakdown of how pointers work: *adjusts cuffs* A pointer simply holds a memory address."
        cleaned = text_and_parsers.clean_and_compact_reply(drifted)
        self.assertFalse(cleaned.lower().startswith("certainly"))
        self.assertIn("memory address", cleaned)

    def test_deterministic_ast_grounding_tools(self):
        msg = "What is 15% of 340 and (24 * 15) + 40? Also convert 100F to C:\n```python\ndef broken(\n```"
        grounding = text_and_parsers.evaluate_deterministic_grounding_tools(msg)
        self.assertIn("15% of 340 = 51", grounding)
        self.assertIn("(24 * 15) + 40 = 400", grounding)
        self.assertIn("37.8°C", grounding)
        self.assertIn("SyntaxError", grounding)


class TestSubconsciousAndMemoryRAG(unittest.TestCase):
    """Tests Step 1.2 (`subconscious_engine.py` & `memory_and_context.py`)."""

    def test_zero_pii_firewall(self):
        # 1. First name allowed; full legal name strips surname
        self.assertEqual(
            subconscious_engine.sanitize_or_reject_sensitive_pii("First name is Alex"),
            "First name is Alex",
        )
        self.assertEqual(
            subconscious_engine.sanitize_or_reject_sensitive_pii("Real name is Alex Mercer"),
            "First name is Alex",
        )
        # 2. Explicit surname blocked
        self.assertIsNone(subconscious_engine.sanitize_or_reject_sensitive_pii("Last name is Smith"))
        # 3. Street address, email, phone, SSN, credit card, API key blocked
        self.assertIsNone(subconscious_engine.sanitize_or_reject_sensitive_pii("Lives at 742 Evergreen Terrace Apt 4B"))
        self.assertIsNone(subconscious_engine.sanitize_or_reject_sensitive_pii("Email is user@example.com"))
        self.assertIsNone(subconscious_engine.sanitize_or_reject_sensitive_pii("Phone number is 555-867-5309"))
        self.assertIsNone(subconscious_engine.sanitize_or_reject_sensitive_pii("Card is 4532 0151 1283 0366"))
        self.assertIsNone(subconscious_engine.sanitize_or_reject_sensitive_pii("API key is sk-" + "proj1234567890abcdefghij"))
        # 4. Broad state/timezone & project facts allowed
        self.assertEqual(
            subconscious_engine.sanitize_or_reject_sensitive_pii("Lives in Texas (CST timezone)"),
            "Lives in Texas (CST timezone)",
        )

    def test_salience_ebbinghaus_retention_and_bm25_rag(self):
        core_f = subconscious_engine.pack_salience_fact(
            "Prefers not to be called sweetie", salience=96, category="BOUNDARY", date_str="2025-01-01"
        )
        sit_f = subconscious_engine.pack_salience_fact(
            "Ate tacos earlier tonight", salience=25, category="SITUATIONAL", date_str="2025-01-01"
        )
        proj_f = subconscious_engine.pack_salience_fact(
            "Builds a custom workstation PC with a Ryzen CPU and dedicated GPU",
            salience=84,
            category="PROJECT",
            date_str="2026-11-10",
        )
        # Core/Boundary (S>=90) never decays
        self.assertGreaterEqual(subconscious_engine.compute_ebbinghaus_retention(core_f), 96.0)
        # Situational fact from a year ago decays near 0
        self.assertLess(subconscious_engine.compute_ebbinghaus_retention(sit_f), 2.0)

        # Eviction drops lowest retention while protecting Core/Boundary
        kept = subconscious_engine.evict_lowest_retention_facts([sit_f, core_f, proj_f], max_items=2)
        self.assertEqual(len(kept), 2)
        self.assertTrue(any("sweetie" in k for k in kept))
        self.assertFalse(any("tacos" in k for k in kept))

        # BM25 + Semantic Tag RAG retrieves clean unprefixed text
        hits = memory_and_context.select_relevant_memories(
            [core_f, proj_f],
            active_context_text="What GPU and CPU are in the custom PC?",
            semantic_tags=["hardware", "ryzen"],
        )
        self.assertTrue(any("Ryzen" in h and not h.startswith("[S:") for h in hits))

    def test_heider_balance_damping_hysteresis_and_zlib_ncd(self):
        # Heider's Structural Balance (#5)
        pos_shift = subconscious_engine.compute_heider_triadic_balance_delta(60, 1.0)
        neg_shift = subconscious_engine.compute_heider_triadic_balance_delta(60, -1.0)
        self.assertGreater(pos_shift, 0)
        self.assertLess(neg_shift, 0)

        # Schmitt-Trigger Hysteresis (#7)
        tier_stay = subconscious_engine.get_disposition_tier_with_hysteresis(48, previous_tier="Trusted Inner Circle")
        self.assertEqual(tier_stay, "Trusted Inner Circle")

        # Kolmogorov Complexity zlib NCD Anti-Loop (#6)
        loop_texts = [
            "*sighs dramatically* Oh look, another mortal asking me about my alchemy ledger in this tavern.",
            "*sighs dramatically* Oh look, another mortal asking me about my alchemy potions in this tavern.",
        ]
        is_loop, ncd_val = subconscious_engine.detect_ncd_structural_loop(loop_texts, threshold=0.42)
        self.assertTrue(is_loop)
        self.assertLess(ncd_val, 0.42)


class TestMediaVisionAndDiscovery(unittest.TestCase):
    """Tests Step 1.3 & Phase 3 (`media_engine.py`, `ai_services.py`, `bot_discovery.py`, `commands_and_embeds.py`)."""

    def test_perceptual_dhash_and_saliency_meme_captions(self):
        img = Image.new("RGB", (240, 240), (25, 45, 85))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(20, 20), (120, 200)], fill=(230, 190, 80))
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        h1 = media_engine.compute_dhash_64(png_bytes)
        self.assertIsNotNone(h1)

        # Re-compress as JPEG quality 70 and verify perceptual dHash distance <= 5
        jpg_buf = BytesIO()
        img.save(jpg_buf, format="JPEG", quality=70)
        h2 = media_engine.compute_dhash_64(jpg_buf.getvalue())
        self.assertLessEqual(media_engine.dhash_hamming_distance(h1, h2), 5)

        # Render meme captions with Laplacian edge-energy saliency placement (#3)
        captioned = media_engine.render_meme_captions(png_bytes, "WHEN THE DHASH HITS", "IN ZERO MILLISECONDS")
        self.assertGreater(len(captioned), 200)

    def test_hawkes_process_two_wave_coordination_and_typo_forgiveness(self):
        now_t = time.time()
        hist = [
            {"role": "user", "author_id": 1, "author_is_bot": False, "content": "yo check this out!", "timestamp": now_t - 5.0},
            {"role": "user", "author_id": 2, "author_is_bot": False, "content": "wait what do you think?", "timestamp": now_t - 2.0},
        ]
        mult = bot_discovery.compute_spontaneous_interest_multiplier(
            clean_content="what do you think?",
            has_media=True,
            is_author_bot=False,
            persona_prompt="You love technology.",
            chat_history_list=hist,
            my_bot_id=99,
            speaker_affinity=55,
        )
        self.assertGreater(mult, 1.8)

        # Damerau-Levenshtein <= 2 typo forgiveness
        self.assertEqual(commands_and_embeds.normalize_command_typos("@uwu !whoam"), "@uwu !whoami")
        self.assertEqual(commands_and_embeds.normalize_command_typos("uwu !relatioships"), "uwu !relationships")

        # Two-Wave Multi-Bot Crowd Cap (3 bots max on collective 'hey bots')
        with tempfile.TemporaryDirectory() as tmpdir:
            scache = os.path.join(tmpdir, "shared_image_cache.json")
            r1 = asyncio.run(bot_discovery.claim_or_coordinate_turn_wave(scache, 555, 10, "Bot1", "hey bots!"))
            bot_discovery.mark_turn_wave_complete(scache, 555, 10)
            r2 = asyncio.run(bot_discovery.claim_or_coordinate_turn_wave(scache, 555, 20, "Bot2", "hey bots!"))
            r3 = asyncio.run(bot_discovery.claim_or_coordinate_turn_wave(scache, 555, 30, "Bot3", "hey bots!"))
            r4 = asyncio.run(bot_discovery.claim_or_coordinate_turn_wave(scache, 555, 40, "Bot4", "hey bots!"))
            self.assertEqual(r1, (True, 0))
            self.assertEqual(r2, (True, 1))
            self.assertEqual(r3, (True, 2))
            self.assertEqual(r4, (False, -1))


class TestMainBotAndOrchestrator(unittest.TestCase):
    """Tests Step 1.4 & Phase 4 (`main_bot.py`, `orchestrator.py`, `web_dashboard.html`)."""

    def test_adaptive_coalescing_and_situational_sampling(self):
        self.assertEqual(main_bot.compute_adaptive_coalesce_delay("How is your project going today?", False), 0.12)
        self.assertEqual(main_bot.compute_adaptive_coalesce_delay("hold on", False), 1.25)

        t_code, p_code = main_bot.compute_situational_sampling_params("Fix this Python traceback:\n```py\nx=1\n```")
        self.assertEqual((t_code, p_code), (0.45, 0.85))
        t_banter, p_banter = main_bot.compute_situational_sampling_params("roast this meme lol skill issue")
        self.assertEqual((t_banter, p_banter), (0.85, 0.92))

    def test_orchestrator_port_scanner_leaky_bucket_and_dashboard(self):
        port = orchestrator.find_free_localhost_port("127.0.0.1", 8765, 8799)
        self.assertTrue(8765 <= port <= 8799)

        limiter = orchestrator.LeakyBucketRestartLimiter(max_tokens=4.0)
        for idx in range(3):
            ok, _ = limiter.record_crash_and_get_backoff(200.0 + idx)
            self.assertTrue(ok)
        ok_fourth, _ = limiter.record_crash_and_get_backoff(205.0)
        self.assertFalse(ok_fourth)
        self.assertTrue(limiter.quarantined)

        ok_ast, msg = orchestrator.validate_python_file_ast("main_bot.py")
        self.assertTrue(ok_ast, msg)
        self.assertTrue(os.path.exists("web_dashboard.html"))



    def test_bot_wizard_env_manager_and_visual_config_editor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_p = os.path.join(tmpdir, ".env")
            orchestrator.upsert_env_variable(env_p, "DEEPINFRA_TOKEN", "di_secret_123456789")
            orchestrator.upsert_env_variable(env_p, "GIPHY_TOKEN", "giphy_secret_987654")
            orchestrator.upsert_env_variable(env_p, "MAX_HISTORY_LENGTH", "250")
            env_map = orchestrator.read_env_file_dict(env_p)
            self.assertEqual(env_map.get("MAX_HISTORY_LENGTH"), "250")

            ok_c, safe_n, py_path = orchestrator.generate_new_bot_script(
                bots_dir=tmpdir,
                env_path=env_p,
                bot_name="Test Butler",
                discord_token="discord_tok_abcdef123456",
                system_prompt_identity="You are a polite butler bot.",
                temperature=0.72,
                keywords=["butler", "steward"],
            )
            self.assertTrue(ok_c, py_path)
            self.assertEqual(safe_n, "test_butler")
            self.assertTrue(os.path.exists(py_path))

            env_ok, missing = orchestrator.verify_bot_script_env_keys(py_path, env_p)
            self.assertTrue(env_ok, f"Unexpected missing keys: {missing}")

            ok_u, msg_u = orchestrator.update_bot_script_constants(
                py_path, {"temperature": 0.84, "enable_voice": True, "keywords": ["butler", "jeeves"]}
            )
            self.assertTrue(ok_u, msg_u)
            inspected = orchestrator.inspect_bot_script_config(py_path)
            self.assertEqual(inspected["temperature"], 0.84)
            self.assertEqual(inspected["enable_voice"], True)
            self.assertEqual(inspected["keywords"], ["butler", "jeeves"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
