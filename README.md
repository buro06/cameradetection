# camwatch

24/7 multi-camera person and face detection with Telegram alerts.

- **Cameras:** USB webcams, RTSP, HTTP(S) MJPEG/HLS streams, JPEG snapshot URLs, and video files (for testing). Any number of cameras can run at once.
- **Person detection:** Ultralytics YOLO11, batched across all cameras on the GPU.
- **Face recognition:** OpenCV Model Zoo **YuNet** (detection) + **SFace** (recognition). Both are fully FOSS (Apache-2.0/MIT) and run through `cv2.dnn`.
- **Telegram alerts** include a 5-second H.264 clip, starting the moment the person is detected. The caption names everyone seen (🟢 trusted, 🟠 known, 🔴 unknown).
- **Trusted people** don't trigger alerts. Unknown faces are saved so you can label them later from Telegram or the CLI.
- **Interactive dashboard** with menus. A headless mode is available for running as a service.

## How alerts are decided

1. YOLO sees a person on **2 consecutive inferences**, which filters out one-frame false positives.
2. Recording starts at the moment the person was **first detected**. A per-camera ring buffer supplies the frames from before the confirmation. Set `events.pre_seconds` to also include time before detection.
3. For the next **5s**, detection and face recognition keep running. Each face observation votes on the identity of its tracked person.
4. When the clip ends:
   - If **everyone** seen is a trusted person (confirmed by at least `trusted_min_matches` consistent face matches), **no alert** is sent.
   - Otherwise an alert is sent, unless every non-trusted person is still in their **per-person, per-camera cooldown** (default 60s).
   - A new or unknown person always alerts, even while someone else is in cooldown.
   - A person whose face is never visible counts as *unknown*, and alerts.

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

Alerts go to `chat_ids`. Commands are accepted only from those chats or from `allowed_user_ids`; everything else is ignored and logged. To keep the token out of `config.yaml`, set it in the `CAMWATCH_TELEGRAM_TOKEN` environment variable instead.

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
| `/people` · `/unknowns` | List known people / resend recent unknown faces for labeling |
| `/name <Name>` · `/trust <Name>` | Reply to a face photo to label it (`/trust` = no alerts for that person) |
| `/name 17 Alice` | Label unknown face #17 without replying |
| `/trust <Name>` · `/untrust <Name>` | Change trust for an existing person |
| `/ignore` | Reply to a face to discard it |

Commands queued while camwatch was offline (older than 5 minutes) are ignored.

## Running 24/7 on Windows

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
| Alerts for people on the street far away | `detection.min_box_height: 0.15` |
| Wrong name assigned | Raise `face.match_threshold` (0.45–0.5) and enroll more varied samples |
| Known person shown as unknown | Enroll more samples (different light and angles), or lower `match_threshold` slightly (not below 0.36) |
| Trusted person still triggers alerts | Their face isn't seen clearly during the 5s window. Lower `face.min_face_px`, raise `events.post_seconds`, or mount the camera at face height |
| Too many alerts for someone lingering | Raise `events.cooldown_seconds` |
| GPU overloaded | `detection.model: yolo11n.pt`, or lower `detection.detect_fps` |

All settings are documented in `config.example.yaml`.

## Development

```bash
pip install -e .[dev]
pytest
```

## Licenses

camwatch uses Ultralytics YOLO (AGPL-3.0), so this project is AGPL-3.0 as well. The face models are OpenCV Zoo YuNet (MIT) and SFace (Apache-2.0).
