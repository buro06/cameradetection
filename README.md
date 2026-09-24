# camwatch

24/7 multi-camera person and face detection with Telegram alerts.

- **Cameras:** USB webcams, RTSP, HTTP(S) MJPEG/HLS streams, JPEG snapshot URLs, and video files (for testing). Any number of cameras can run at once.
- **Person detection:** Ultralytics YOLO11, batched across all cameras on the GPU.
- **Face recognition:** OpenCV Model Zoo **YuNet** (detection) + **SFace** (recognition). Both are fully FOSS (Apache-2.0/MIT) and run through `cv2.dnn`.
- **Telegram alerts** include a 5-second H.264 clip, starting the moment the person is detected. The caption names everyone seen (🟢 trusted, 🟠 known, 🔴 unknown).
- **Trusted people** don't trigger alerts. Unknown faces are saved so you can label them later from Telegram or the CLI.
- **Interactive dashboard** with menus. A headless mode is available for running as a service.

## How alerts are decided

1. YOLO sees a person on **2 inferences**, which filters out one-frame false positives. A box that appears *on top of* someone already tracked must last about **1 second** (`detection.overlap_confirm_seconds`) before it counts as another person. YOLO sometimes boxes the same person twice for a split second, and this stops that from turning into a phantom "Unknown person".
2. Recording starts at the moment the person was **first detected**. A per-camera ring buffer supplies the frames from before the confirmation. Set `events.pre_seconds` to also include time before detection.
3. For the next **5s**, detection and face recognition keep running. Each face observation votes on the identity of its tracked person.
4. When the clip ends, only the people who are **new** in this clip are considered. Everyone already present is listed in the caption for context but can't cause an alert.
   - If all the new people are trusted (confirmed by at least `trusted_min_matches` consistent face matches), **no alert** is sent.
   - Otherwise an alert is sent. The exception is a known person or known unknown face that **left and came back** within `events.cooldown_seconds` (default 60s) on that camera.
   - A person whose face is never visible counts as *unknown* and alerts. Each such person is tracked separately, so a second stranger always alerts.
   - A quick visit alerts too, even if the person leaves before the clip ends.

**People who stay are never re-alerted.** A group sitting on the couch alerts once, when they arrive.
- **Briefly hidden people:** someone who was sitting or standing still and vanishes for a while (someone walks in front, they lean out of view) is re-attached to their original track when they reappear in the same spot. This applies for up to `events.lost_memory_seconds`, default 30s.
- **People who leave:** someone who was *moving* when they disappeared is treated as gone. A newcomer stepping into that spot or through the same doorway is always treated as new.
- **Trusted people who look away:** once confirmed, a trusted person stays trusted while they're tracked, even when their face is turned away. The exception is when a *different* known person's face starts to outvote them.

If a household member walks in with their face turned away, you get one "Unknown person (face not visible)" alert. Nothing further comes once they sit down.

Several safety rules keep a trusted face from silencing an alert for someone else:
- A face only counts for the person whose head position it matches.
- A face that could belong to either of two overlapping people is discarded.
- A trusted name must win the majority of face observations on that track.

## Install on Windows (Quadro M4000 / Maxwell)

The Quadro M4000 is a Maxwell GPU (compute capability 5.2). PyTorch's CUDA 12.8+ builds dropped Maxwell, so **use the cu126 build**. Install PyTorch *before* everything else:

```powershell
# Python 3.12 from python.org, latest NVIDIA Quadro driver (the R580 branch is the last with Maxwell support)
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .
camwatch doctor
```

`camwatch doctor` should report `CUDA — Quadro M4000 (sm_52)` and `detector device: cuda:0`. If it reports *"not supported by torch …"*, a CUDA 12.8+/13 build of torch was installed. Reinstall with the cu126 command above.

On macOS or Linux: `python -m venv .venv && . .venv/bin/activate && pip install -e .` (CPU, CUDA, or Apple MPS are picked automatically).

## First run

```powershell
camwatch setup      # Telegram bot + cameras wizard (writes config.yaml)
camwatch            # start monitoring with the live dashboard
```

**Telegram:**
1. Create a bot with **@BotFather** (`/newbot`).
2. Paste the token into the setup wizard.
3. Send `/start` to your bot, or add it to a group.
4. Choose **Find chats & users**.

Alerts go to `chat_ids`. `allowed_user_ids` is the command whitelist. When it's set, only those users can use commands and the face-labelling buttons, in any chat, alert groups included, so other group members still get alerts but can't control the bot. When it's empty, anyone in an alert chat can. Everything else is ignored and logged. To keep the token out of `config.yaml`, set it in the `CAMWATCH_TELEGRAM_TOKEN` environment variable instead.

### Dashboard keys

| Key | Action |
|---|---|
| `m` | Menu: cameras, people & faces, review unknown faces, arm/disarm, Telegram, settings |
| `a` | Arm/disarm all cameras |
| `s` | Save a snapshot of every camera to `data/snapshots/` |
| `q` | Quit |

Monitoring keeps running while menus are open.

### Adding faces

- **Live capture:** Menu → People & faces → *Enroll a person from a live camera*. The person stands alone about 1–2 m from the camera and slowly turns their head. About 12 varied samples are captured, with an optional preview window.
- **From alerts (Telegram):** each unknown face arrives as a photo with buttons for your known people. Either tap a name, or reply to the photo with `/name Alice` or `/trust Alice`.
- **Review in CLI:** Menu → *Review unknown faces*. Similar unknown faces are grouped into clusters, and each cluster opens in your image viewer so you can name it, trust it, or delete it.

### Telegram commands

| Command | |
|---|---|
| `/status` | Camera health, arm state, what's in view, last event |
| `/arm [camera]` · `/disarm [camera]` | Enable or pause alerts (all cameras, or one) |
| `/snapshot [camera]` | Live picture(s) with detections drawn |
| `/disk` | Space used by the camwatch folder (clips, snapshots, faces, models, logs, Python env) vs. free and total disk space, with a low-space warning |
| `/people` · `/unknowns` | List known people / resend recent unknown faces for labeling |
| `/name <Name>` · `/trust <Name>` | Reply to a face photo to label it (`/trust` = no alerts for that person) |
| `/name 17 Alice` | Label unknown face #17 without replying |
| `/trust <Name>` · `/untrust <Name>` | Change trust for an existing person |
| `/ignore` | Reply to a face to discard it |

Commands queued while camwatch was offline (older than 5 minutes) are ignored.

## Running 24/7 on Windows

**Simplest: the watcher script.** Double-click `run-camwatch.cmd` in the camwatch folder, or run it from a cmd window. It activates `.venv`, starts camwatch, and restarts it in the same window whenever it exits, whether it crashed or you pressed `q`:

```bat
run-camwatch                  :: live dashboard
run-camwatch run --headless   :: no dashboard
```

After camwatch exits, the watcher waits 10 s before restarting. Press `R` to restart now or `Q` to stop. If camwatch keeps failing within a minute of starting, the wait grows up to 5 minutes. Restarts are logged to `data\logs\watcher.log`. To start the watcher at log on, put a shortcut to it in `shell:startup`.

**Option A: NSSM service** (starts at boot, restarts on crash):

```powershell
nssm install camwatch "C:\camwatch\.venv\Scripts\camwatch.exe" "--config C:\camwatch\config.yaml run --headless"
nssm set camwatch AppDirectory C:\camwatch
nssm set camwatch AppEnvironmentExtra CAMWATCH_TELEGRAM_TOKEN=123456:ABC...
nssm start camwatch
```

A service can't open USB webcams under some drivers. If yours can't, use option B.

**Option B: Task Scheduler.** Create a task that runs *at log on*:
- Program: `C:\camwatch\.venv\Scripts\camwatch.exe`
- Arguments: `--config C:\camwatch\config.yaml`
- Settings: "restart every 1 minute" on failure

This option keeps the interactive dashboard.

Logs are written to `data/logs/camwatch.log` (rotated), every event to `data/events.jsonl`, and alert clips to `data/clips/<date>/` (deleted after `clip.retention_days`).

## Tuning

| Symptom | Setting |
|---|---|
| USB webcam stuck at 640×360 / 640×480 on Windows | Menu → Cameras → *camera* → **Resolution** (e.g. 1920x1080). If it still doesn't change, try **Pixel format / capture backend** → MJPG, or backend msmf. The Cameras table shows `640x360 (asked 1920x1080)` when the camera ignores the request |
| USB webcam stuck at ~5 fps at 1080p | Menu → Cameras → *camera* → **Frame rate** → 30 fps (or `fps: 30` under the camera in `config.yaml`). The log line `USB camera opened: 1920x1080 MJPG @ 30 fps` shows what the camera accepted. A warning after 5 s means frames still arrive slowly. If the format shown is YUY2, set **Pixel format** → MJPG or try backend msmf; if it's MJPG, the room is probably too dim |
| Alerts for people on the street far away | `detection.min_box_height: 0.15` |
| Wrong name assigned | Raise `face.match_threshold` (0.45–0.5) and enroll more varied samples |
| Known person shown as unknown | Enroll more samples (different light and angles), or lower `match_threshold` slightly (not below 0.36) |
| Trusted person still triggers alerts | Their face isn't seen clearly during the 5s window. Lower `face.min_face_px`, raise `events.post_seconds`, or mount the camera at face height |
| Duplicate boxes on one person / phantom "Unknown person (face not visible)" alerts | Raise `detection.overlap_confirm_seconds` (e.g. 2) |
| Same person alerts again after briefly stepping out | Raise `events.cooldown_seconds` (only applies to recognised faces / unknown-face clusters) |
| Someone who stays still gets "new" alerts after being hidden a long time | Raise `events.lost_memory_seconds` |
| GPU overloaded | `detection.model: yolo11n.pt`, or lower `detection.detect_fps` |

All settings are documented in `config.example.yaml`.

## Development

```bash
pip install -e .[dev]
pytest
```

## Licenses

camwatch uses Ultralytics YOLO (AGPL-3.0), so this project is AGPL-3.0 as well. The face models are OpenCV Zoo YuNet (MIT) and SFace (Apache-2.0).
