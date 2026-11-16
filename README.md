Multi-Agent Discord AI Ecosystem
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Discord.py](https://img.shields.io/badge/Discord.py-Asynchronous-5865F2?style=for-the-badge&logo=discord&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Video_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-Meme_Engine-3776AB?style=for-the-badge)
![DeepInfra](https://img.shields.io/badge/DeepInfra-Multimodal_AI-000000?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
A distributed multi-agent Discord AI framework and SRE process control plane written in Python. Engineered to run 4 to 50+ autonomous, distinct AI personas concurrently within the same Discord channels, this system features an OS-level process supervisor and real-time Server-Sent Events (SSE) Web-UI dashboard, a two-phase Conscious/Subconscious cognitive architecture, Okapi BM25 + Ebbinghaus exponential decay RAG, Heider structural balance social graph dynamics, two-wave multi-agent turn coordination, 64-bit perceptual dHash computer vision with Shannon-entropy/action-flux 2x2 video storyboards, Laplacian edge-energy meme typography, and bidirectional voice (OpenAI Whisper STT + MiMo-V2.5-TTS Director Mode).
(This was made with the major help of Gemini Argon)
---
Key Engineering Highlights
SRE Process Orchestrator & Real-Time Web-UI Control Plane (`orchestrator.py`, `web\_dashboard.html`):
Zero-Dependency `asyncio` HTTP/SSE Server & Port Auto-Scanner: Serves a single-file dark-mode cognitive control plane on `127.0.0.1:8765` (automatically probing ports `8765..8799` in $<1\text{ ms}$ to avoid collisions with standard web-dev ports `3000`, `5173`, `8000`, `8080`). Streams live IPC logs, telemetry, and subconscious JSON payloads over `text/event-stream`.
Leaky Token-Bucket Restart Guard & Orphaned Process Reaper: Models child process stability via $B_t = \min(B_{\max}, B_{t-1} + r \cdot \Delta t) - 1$ with full-jitter exponential backoff ($t_{\text{sleep}} = \min(300, 2^k + U(0, 1.5))$), tripping a `QUARANTINED` circuit breaker after 4 crashes in $90\text{ s}$. Automatically sweeps and terminates detached orphan processes and stale `.lock`/`.tmp` files on startup and stop.
AST-Validated Zero-Downtime Rolling Hot-Reload: Monitors `.py` file `st\_mtime\_ns` timestamps across `python\_modules/` and `bots/`. On file save, validates syntax via `ast.parse()` and `py\_compile` before executing a staggered $2.0\text{ s}$ rolling restart across active agents—aborting the reload if syntax errors are detected so running bots never crash.
Native OS RAM/CPU Profiler, FinOps Ledger & Web-UI Bot Studio: Polls child PID Working Set RAM (MB) and CPU % every $3\text{ s}$ using native Win32 `ctypes` (`GetProcessMemoryInfo` / `GetProcessTimes`) and POSIX `/proc` APIs with zero `psutil` dependency. Includes an in-browser Bot Studio (create new bots, edit constants/prompts with AST validation, and manage `.env` keys), a Fruchterman-Reingold Force-Directed Social Graph Canvas, and a 1-click Public HTTPS Tunnel Manager (`cloudflared` / `ssh`) gated by constant-time `WEBUI\_ADMIN\_TOKEN` authentication.
Cross-Process Concurrency, Dead-PID Lock Recovery & EWMA Circuit Breakers (`core\_system.py`, `main\_bot.py`):
OS-Agnostic Atomic File Locking: Built a cross-process synchronization engine (`FileLock`) using atomic `O\_CREAT | O\_EXCL` lockfiles, monotonic timeout tracking, and atomic `.tmp` file replacement (`os.replace`). Inspects lockfile `<pid>|<timestamp>` headers against the OS process table (`OpenProcess`/`GetExitCodeProcess` on Windows; `os.kill(pid, 0)` on POSIX) to evict locks held by terminated processes in $0\text{ ms}$.
Zero-Copy Serialization & Nanosecond RAM Cache: Enables independent bot processes to read and mutate shared JSON state (`chat\_memory.json`, `shared\_image\_cache.json`, `general\_knowledge.json`) concurrently without race conditions. Backed by a thread-safe (`threading.RLock`) `(st\_mtime\_ns, st\_size)` RAM cache (`0.001ms` reads) with non-blocking `asyncio.to\_thread` I/O wrappers.
Velocity-Aware Adaptive Coalescing & Situational Sampling: Dynamically adjusts message coalescing windows (`0.12s` fast-path for complete punctuated sentences $\ge 5$ words, saving $\sim 730\text{ ms}$ of typing latency, vs. `1.25s` for short unpunctuated fragments) and modulates `(temperature, top\_p)` per turn (`(0.45, 0.85)` for Technical/Code/Math precision vs. `(0.85, 0.92)` for creative banter).
EWMA Latency Tracker & Adaptive Circuit Breaker: Tracks Exponentially Weighted Moving Average latency ($L_t = 0.3 \cdot t_{\text{obs}} + 0.7 \cdot L_{t-1}$) and error rates per `(endpoint, model)` pair, sorting fallback models by lowest effective latency $L_t / (1 - P_{\text{err}})$ when a primary endpoint trips its circuit breaker.
Two-Wave Multi-Agent Coordination, Hawkes Process Engagement & Discovery (`bot\_discovery.py`):
Two-Wave Co-Star Micro-Batching & Crowd Cap (10–50 Bot Scale): Caps collective greetings (`"hey bots"`) at 3 responders max. When multiple agents are addressed (`"hey guild assistant and cat bot"`), Wave 1 (`slot 0`) generates immediately ($0\text{ ms}$ wait) while Wave 2 Co-Stars (`slot >= 1`) wait in parallel on Wave 1 ($\le 2.2\text{ s}$ max), reload shared history to read Wave 1's remark, and generate simultaneously—preventing $N \times 2.5\text{ s}$ serial latency stacking.
Self-Exciting Hawkes Process & Progressive Pity Timer: Models human message arrival intensity via $\lambda(t) = \sum \alpha e^{-\beta (t - t_k)}$ ($\alpha = 0.18, \beta = 0.035\text{ s}^{-1}$) to boost spontaneous engagement during rapid multi-user banter, paired with a $+1.5%/\text{msg}$ progressive pity timer since the agent last spoke.
Possessive-Safe Longest-Match Disambiguation & 4-Stage Follow-Up Resolver: Disambiguates multi-agent addressing using possessive-safe boundaries (`\_term\_boundary\_pattern`) and longest-match-first token subsumption. Combines $O(1)$ tombstone/turn-cap checks (Stage 0), deterministic addressing/dismissal filters (Stage 1), stem-aware lexical/visual overlap (Stage 2), and a fast `Qwen3-30B-A3B` classifier (Stage 3).
Two-Phase Subconscious Cognitive Engine, Okapi BM25 + Ebbinghaus RAG & Social Graph (`subconscious\_engine.py`, `memory\_and\_context.py`):
Two-Phase Brain Split & Hidden `\[THOUGHT:]` Subtext: Phase A (Conscious Reflex) strips hidden `\[THOUGHT: ...]` inner-monologue reasoning, injects $0\text{ ms}$ AST math/unit/Python syntax grounding tools, and dispatches the Discord reply immediately. Phase B (Subconscious Reflection) runs asynchronously post-send (`asyncio.create\_task`) on `Qwen/Qwen3-30B-A3B` under a single-leader per-message lock (`\_subconscious\_leader\_msg`) to extract user facts, server lore, rapport shifts, room state, 5-turn TTL room props, and episodic session chapters in a single batched JSON call.
1–100 Salience Spectrum, Okapi BM25 & Ebbinghaus Forgetting Curve: Tags memories with `\[S:XX|CAT|YYYY-MM-DD|R:N]` across `CORE`/`BOUNDARY` ($S \ge 90$, $\tau = \infty$, immune to eviction), `PROJECT`/`PREF`/`REL` ($\tau = 120\text{d}$), `HABIT`/`LORE` ($\tau = 35\text{d}$), and `SITUATIONAL` ($\tau = 7\text{d}$). Computes retention via:
$$R(t) = S \cdot \exp\left(-\frac{\Delta t_{\text{days}}}{\tau_{\text{cat}} \cdot (1 + 0.4 \cdot C_{\text{recall}})}\right)$$
and retrieves topically relevant memories in $<1\text{ ms}$ using Okapi BM25 ($k_1 = 1.5, b = 0.75$) + semantic tag expansion.
Heider's Structural Balance Theory & Schmitt-Trigger Hysteresis: Models server relationships as a signed directed graph $G = (V, E, w)$ where $w_{ij} \in [-100, +100]$. Evaluates triadic rapport shifts via $\Delta w_{i \to A} = \text{round}(8.0 \cdot \tanh(w_{i \to B} / 45.0) \cdot \text{Valence}(A \to B))$, applies $1/\sqrt{k}$ session diminishing marginal returns, and stabilizes Disposition Tier transitions with a $\pm 3$-point Schmitt-Trigger hysteresis band.
Kolmogorov Complexity (`zlib` NCD) Anti-Loop & Token-Budgeted Context Slicer: Computes Normalized Compression Distance $\text{NCD}(x, y) = \frac{C(x \cdot y) - \min(C(x), C(y))}{\max(C(x), C(y))}$ across recent bot replies in $0.04\text{ ms}$ (injecting a structural pivot directive if $\text{NCD} < 0.42$) and enforces a $\sim 3,500$-token dynamic context budget.
Perceptual dHash Vision, Action-Flux/Entropy Storyboards, Saliency Memes & Bidirectional Voice (`media\_engine.py`, `ai\_services.py`):
64-Bit Perceptual dHash Vision Cache: Computes a $9 \times 8$ horizontal gradient bitmask (`compute\_dhash\_64`) and checks bitwise XOR Hamming distance $\le 5$ so reposted or re-compressed memes, stickers, and GIFs resolve from cache in $0.1\text{ ms}$ with zero Vision API calls.
Temporal Action-Flux ($L_1$) & Shannon Entropy 2x2 Storyboarding: Samples candidate frames across videos (`.mp4`, `.webm`, `.mov`) and animated GIFs/WebPs under a bounded semaphore (`BoundedSemaphore(2)`), scoring Shannon entropy $H = -\sum p_k \log_2 p_k$, Laplacian sharpness, and temporal motion flux $|I_t - I_{t-1}|_1$ to stamp peak action keyframes (`1`–`4`) onto a 2x2 storyboard.
Image-to-Image Style Remixing & Laplacian Edge-Energy Saliency Memes: Supports natural image remixing (`"remix this in cyberpunk style"`) by blending Vision descriptions into FLUX prompts, attaches interactive `\[🎨 View Prompt]` and `\[🔄 Redraw]` buttons, and measures 2D discrete Laplacian edge energy in top vs. bottom bands so single-caption memes never obscure high-detail subject faces.
Bidirectional Voice (Whisper STT + MiMo-V2.5-TTS): Transcribes human Discord voice notes (`.ogg`, `.mp3`, `.wav`, `.m4a`) via `openai/whisper-large-v3-turbo` and synthesizes expressive replies with `MiMo-V2.5-tts-voicedesign`.
Interactive Discord UI Hub (`/hub`), Fuzzy Typo Forgiveness & Zero-PII Security Firewall (`commands\_and\_embeds.py`):
Single Clutter-Free `/hub` Slash Command & Damerau-Levenshtein ($\le 2$) Typo Correction: Registers an instant guild-synced `/hub` slash command (and auto-heals text command typos like `!whoam` $\rightarrow$ `!whoami`) that opens an interactive `discord.ui.View` card with `\[🗂️ Dossier]`, `\[🌐 Social Graph]`, `\[📖 Chapters]`, `\[✏️ Manage My Facts]`, and `\[⚙️ Admin Panel]` tabs.
Strict Caller Ownership Isolation & $0\text{ ms}$ Zero-PII Firewall: Binds personal fact management strictly to `interaction.user.display\_name` (preventing users from editing another member's profile) and scrubs all candidate facts through `sanitize\_or\_reject\_sensitive\_pii`—allowing first names while redacting surnames from full-name strings and blocking street addresses, emails, phone numbers, SSNs, credit cards, and API keys.
---
System Architecture
```text
+===================================================================================+
|         SUPERVISOR \& REAL-TIME WEB-UI CONTROL PLANE (orchestrator.py)             |
|  • Auto-Port http://127.0.0.1:8765+  | Built-In HTTPS Tunnel (cloudflared / ssh)  |
|  • Leaky-Bucket Restart Guard        | AST Rolling Hot-Reload | Ghost PID Reaper  |
|  • Live SSE Dashboard (web\_dashboard.html):                                       |
|     ├─ Force-Directed Social Graph   | Subconscious \[THOUGHT:] X-Ray Stream       |
|     └─ Ebbinghaus Salience CRUD      | Bot Studio Editor, .env \& FinOps Matrix    |
+=========================================+=========================================+
                                          | (Spawns, Monitors \& Streams IPC)
                                          v
+-----------------------------------------------------------------------------------+
|                                  DISCORD SERVER                                   |
|   /hub Slash Command, @Mentions, Replies, Voice Notes, Videos, Images \& Reactions |
+-----------------------------------------+-----------------------------------------+
                                          |
                   +----------------------+----------------------+
                   |                      |                      |
                   v                      v                      v
        +--------------------+ +--------------------+ +--------------------+
        |  guild\_assistant   | |      cat\_bot       | |  custom\_agent\_01   |
        |   (Process #1)     | |   (Process #2)     | |    (Process #3)    |
        +---------+----------+ +----------+---------+ +----------+---------+
                  |                       |                      |
                  +-----------------------+----------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     MODULAR PYTHON ENGINE (bot\_info/python\_modules)               |
|                                                                                   |
|  +-----------------------+  +-----------------------+  +-----------------------+  |
|  |      main\_bot.py      |  |   bot\_discovery.py    |  | subconscious\_engine.py|  |
|  | Adaptive coalescing,  |  | Two-Wave coordinator, |  | Async 30B JSON turn   |  |
|  | situational sampling, |  | Hawkes process chime, |  | harvester, BM25 +     |  |
|  | \& Whisper/FLUX/TTS    |  | \& 4-stage follow-ups  |  | Ebbinghaus \& Zero-PII |  |
|  +-----------------------+  +-----------------------+  +-----------------------+  |
|                                                                                   |
|  +-----------------------+  +-----------------------+  +-----------------------+  |
|  | memory\_and\_context.py |  |    media\_engine.py    |  |    ai\_services.py     |  |
|  | Heider balance graph, |  | 64-bit dHash, Entropy |  | Perceptual dHash hit, |  |
|  | Schmitt hysteresis,   |  | \& Action-Flux 2x2     |  | image remixing, FLUX, |  |
|  | \& zlib NCD anti-loop  |  | video \& saliency meme |  | Whisper STT \& MiMo TTS|  |
|  +-----------------------+  +-----------------------+  +-----------------------+  |
|                                                                                   |
|  +-----------------------+  +-----------------------+  +-----------------------+  |
|  |commands\_and\_embeds.py |  |  text\_and\_parsers.py  |  |    core\_system.py     |  |
|  | /hub slash card, UI   |  | \[THOUGHT:] stripper,  |  | Dead-PID FileLock,    |  |
|  | modals, typo fixer \&  |  | Drift Guard \& 0ms AST |  | EWMA latency failover,|  |
|  | caller-isolated CRUD  |  | math/unit/code tools  |  | RAM cache \& HTTP pool |  |
|  +-----------------------+  +-----------------------+  +-----------------------+  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                  ATOMIC SHARED \& PER-BOT PERSISTENCE (JSON + FileLock)            |
|                                                                                   |
|  \[Shared Across All Agents]                 \[Isolated Per-Agent Persona]          |
|  - chat\_memory.json (In-Place Reactions)    - <agent>\_long\_term\_memory.json       |
|  - shared\_image\_cache.json (dHash \& State)  - <agent>\_user\_profiles.json          |
|  - general\_knowledge.json (Universal Facts) - <agent>\_channel\_configs.json        |
+-----------------------------------------------------------------------------------+
```
---
Repository Structure
```text
multi-agent-discord-framework/
├── bots/
│   ├── guild\_assistant.py                 # Reference steward \& quartermaster AI agent
│   ├── cat\_bot.py                         # Reference non-verbal / expressive companion AI agent
│   ├── custom\_agent\_alpha.py              # Additional autonomous persona script
│   └── bot\_info/
│       ├── .env                           # API keys, Discord tokens \& WEBUI\_ADMIN\_TOKEN (gitignored)
│       ├── bot\_logs/                      # Per-agent real-time execution logs
│       ├── config\_files/                  # Per-channel runtime settings JSONs
│       ├── long\_term\_memory/              # Per-agent lore, user dossiers \& shared general facts
│       ├── short\_term\_memory/             # Shared channel history, dHash vision cache \& room state
│       └── python\_modules/
│           ├── orchestrator.py            # OS process supervisor, FinOps profiler \& HTTP/SSE server
│           ├── web\_dashboard.html         # High-contrast OLED real-time cognitive control plane UI
│           ├── main\_bot.py                # Core Discord client lifecycle, /hub sync \& turn pipeline
│           ├── subconscious\_engine.py     # Post-send 30B JSON analyzer, BM25/Ebbinghaus \& Zero-PII
│           ├── memory\_and\_context.py      # Heider social graph, NCD anti-loop \& context assembly
│           ├── commands\_and\_embeds.py     # /hub slash card, interactive UI views/modals \& commands
│           ├── ai\_services.py             # Perceptual dHash cache, image remix, FLUX, Whisper \& TTS
│           ├── bot\_discovery.py           # Two-Wave coordinator, Hawkes process \& follow-up resolver
│           ├── media\_engine.py            # OpenCV/PIL entropy storyboards, saliency memes \& dHash
│           ├── text\_and\_parsers.py        # \[THOUGHT:] \& TTS parser, Drift Guard \& 0ms AST tools
│           ├── core\_system.py             # Dead-PID FileLock, EWMA latency tracker \& RAM cache
│           ├── config.py                  # Pre-compiled regexes, salience constants \& fallbacks
│           └── test\_framework.py          # Offline unit \& integration test suite (12/12 tests)
├── manage\_bots.bat                        # 1-Click Windows Orchestrator \& Web-UI Launcher
├── manage\_bots.sh                         # 1-Click POSIX/Linux/macOS Orchestrator \& Web-UI Launcher
├── universal\_copy\_paste\_bot\_template.py   # Drop-in template for spinning up new AI personas
└── LICENSE
```
---
Getting Started
1. Prerequisites
Python 3.10+
API Keys:
One or more Discord Bot Tokens (with `Message Content Intent` and `Server Members Intent` enabled in the Discord Developer Portal).
A DeepInfra API Key (for LLM text, Vision, Whisper STT, FLUX image generation, and MiMo TTS).
(Optional) A Giphy API Key (for reaction GIF search).
2. Installation
Clone the repository and install the required Python dependencies:
```bash
git clone https://github.com/disciplehazen/multi-agent-discord-framework.git
cd multi-agent-discord-framework
pip install discord.py requests python-dotenv Pillow opencv-python
```
3. Launching the Supervisor & Real-Time Web-UI Control Plane
Launch the unified process orchestrator and Web-UI dashboard using the included launcher scripts:
Windows:
```cmd
  manage\_bots.bat
  ```
Linux / macOS:
```bash
  chmod +x manage\_bots.sh
  ./manage\_bots.sh
  ```
Run Offline Unit Test Suite:
```bash
  ./manage\_bots.sh --test
  ```
Once launched:
Open `http://127.0.0.1:8765` in your browser (all discovered bots start in `STOPPED` standby mode by default so you can choose which agents to launch, or pass `--autostart` to start all agents on boot).
Open the `🔑 API Keys \& .env` tab to save your `DEEPINFRA\_TOKEN`, `GIPHY\_TOKEN`, and bot tokens into `bots/bot\_info/.env`.
Open the `🛠️ Bot Studio` tab to create new bots (`\[C] Create Bot Wizard`), edit model constants/prompts (`\[E] Visual Config Editor`), or start/restart individual bots from the `🖥️ Fleet \& FinOps` tab.
(Optional Remote Admin Access) Click `START PUBLIC TUNNEL` in the Web-UI header to provision a secure HTTPS tunnel, or type `@BotName admin webui` in Discord to receive a private 1-click magic login link via DM.
---
In-Chat Command Suite
Commands can be triggered via the `/hub` slash command, `@BotName` mention, direct reply to a bot's message, keyword prefix (e.g., `steward !whoami` or `kitty !chapters`), or multi-bot broadcast (`!bots status`). Damerau-Levenshtein ($\le 2$) fuzzy matching automatically corrects typos like `!whoam` or `!chaptrs`.
User Commands & Interactive Hub
Command	Description
`/hub \[target] \[private]`	Opens the all-in-one interactive Dossier Hub card with buttons for `\[🗂️ Dossier]`, `\[🌐 Social Graph]`, `\[📖 Chapters]`, caller-isolated `\[✏️ Manage My Facts]`, and `\[⚙️ Admin Panel]`.
`@BotName !whoami \[name]`	Opens the interactive Dossier Hub & Affinity progress bar (`-100` to `+100`) for yourself or another member/bot.
`@BotName !relationships`	Displays the agent's complete Server Social Graph (`Trusted Inner Circle`, `Friendly Allies`, `Comedic Foils`, and `Rivals`).
`@BotName !chapters`	Displays rolling Episodic Session Chapters (storyline recaps synthesized by the Subconscious Engine).
`@BotName !forgetme <keyword|all>`	Removes specific matching facts (or wipes `all` personal facts) strictly from your own profile.
`!bots status` / `!bots whoami`	Broadcasts a status or dossier check across all active bots in the channel simultaneously.
`@BotName !help`	Displays the complete in-chat command reference.
Administrator Channel, Web-UI & Memory Controls
Command	Description
`@BotName admin webui`	Sends a verified Server Administrator a private 1-click magic login link to the live Web-UI Control Plane via DM.
`@BotName admin status`	Shows channel configuration, active media flags, welcome greeter state, and memory counts.
`@BotName admin enable` / `disable`	Enables or disables the agent in the current channel.
`@BotName admin welcome <on|off|test>`	Enables, disables, or previews in-character server-join welcome messages in the current channel.
`@BotName admin reactonly on` / `off`	Toggles silent emoji-reaction-only mode in the current channel.
`@BotName admin chances <resp> <emoji>`	Sets baseline spontaneous chime-in (`0.0–1.0`) and emoji reaction (`0.0–1.0`) probabilities.
`@BotName admin random <minutes>`	Sets the unprompted periodic post timer in minutes (`0` = disabled).
`@BotName admin temp <0.1-1.5>`	Adjusts the baseline LLM creativity temperature for the current channel.
`@BotName admin features <type> <on|off>`	Toggles `vision`, `gifs`, `image` (images & memes), or `voice` per channel.
`@BotName admin memory`	Displays a salience-tagged snapshot of learned Server Lore, Shared General Facts, and User Profiles.
`@BotName admin learn <fact>`	Manually adds or consolidates a server-specific lore entry (protected by the Zero-PII Firewall).
`@BotName admin learngeneral <fact>`	Manually adds or consolidates a shared universal knowledge fact across all agents.
`@BotName admin learnuser <user> | <fact>`	Manually adds or consolidates a fact in a specific user's or bot's dossier.
`@BotName admin forget <keyword>`	Purges any server lore, general fact, or user dossier entry matching `<keyword>`.
`@BotName !clearmemory`	Wipes short-term conversation history and recent generated media cache for the current channel.
---
Creating Custom Personas
You can spin up new autonomous agents with full access to shared perceptual vision, custom meme generation, voice synthesis, and cross-agent discovery in two ways:
Via the Web-UI Bot Studio (Recommended): Open `http://127.0.0.1:8765` $\rightarrow$ `Bot Studio` tab, enter the Bot Name, Discord Token, Trigger Keywords, and System Prompt Identity, and click `Create Bot \& Start Immediately`. The supervisor automatically writes `bots/<name>.py`, registers `<NAME>\_DISCORD\_TOKEN` in `bots/bot\_info/.env`, validates the AST, and spawns the agent.
Via Template Script: Copy `universal\_copy\_paste\_bot\_template.py` into `bots/` (e.g., `bots/archivist\_agent.py`) and customize `TOKEN`, `TRIGGER\_KEYWORDS`, and `get\_system\_prompt()`.
---
License
Distributed under the MIT License. See `LICENSE` for full details.