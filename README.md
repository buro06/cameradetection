# camwatch

24/7 multi-camera person and face detection with Telegram alerts.

- **Cameras:** USB webcams, RTSP, HTTP(S) MJPEG/HLS streams, JPEG snapshot URLs, and video files (for testing). Any number of cameras can run at once.
- **Person detection:** Ultralytics YOLO11, batched across all cameras on the GPU.
- **Face recognition:** InsightFace **buffalo_l**: **SCRFD-10G** (detection) + **ArcFace R50** (recognition), run through onnxruntime (GPU if it works, otherwise CPU). The model weights are free for non-commercial use only, e.g. at home.
- **Telegram alerts** include a 5-second H.264 clip, starting the moment the person is detected. The caption names everyone seen (🟢 trusted, 🟠 known, 🔴 unknown).
- **Trusted people** don't trigger alerts. Unknown faces are saved so you can label them later from Telegram or the CLI.
- **Hard cooldown:** each camera sends at most one Telegram alert per 30s (`telegram.min_alert_interval_seconds`); later ones are still clipped and logged as `rate-limited`.
- **Interactive dashboard** with menus. A headless mode is available for running as a service.

## How alerts are decided

camwatch alerts on **arrivals**, not on presence. It doesn't follow individual bodies from frame to frame, so walking around, sitting down, turning away or a missed detection can't turn one person into two. It combines two signals.

**People count (YOLO).**
- Only solid person boxes count. A box must be at least `detection.person_min_height` (15%) of the frame height, or twice that if it's cut off at the frame edge, so an arm or shoulder poking into view doesn't count. A box inside a bigger one is the same person boxed twice.
- The count goes **up** once more people have been in view for `events.arrive_seconds` (2s). That's an **arrival**.
- The count goes **down** only once fewer people have been in view for `leave_seconds`. The default is 60s, for living rooms; set about 10s per camera for outdoor cameras people walk past (menu → Cameras → *camera* → **Leave time**, or `leave_seconds:` under the camera).

**Faces (InsightFace).**
- Every clear face in the frame is identified, wherever it is: a known person, or grouped with similar unknown faces.
- Blurry or turned-away faces are ignored (`face.quality_min_sharpness`, `face.quality_max_turn`).
- A face must be seen clearly at least `face.min_sightings` times (2) during a clip to count.

**Activity sessions.** A session starts when someone appears in an empty scene and lasts until nobody has been in view for `leave_seconds`. Inside a session nothing re-alerts, except:
- **more people** than before (the count rises again), or
- **a new face** that hasn't been seen in this session.

**Each event:**
1. Starts on an arrival or a new face. The clip begins when the person first appeared (plus `events.pre_seconds`).
2. Runs for `events.post_seconds` (5s) while faces keep being identified.
3. Lists the people:
   - everyone whose face was seen enough
   - for each arrival that no newly seen face accounts for, a **"Person (face not visible)"** entry (mask, hood, dark room, back turned)
4. Decides:
   - no alert if every new person is trusted
   - otherwise an alert, unless the same face already alerted on this camera within `events.cooldown_seconds` (60s)
5. An alert still isn't *sent* if the same camera sent one less than `telegram.min_alert_interval_seconds` (30s) ago. It's clipped and logged as `rate-limited`.

**Known limits:**
- A household member who leaves the room (longer than `leave_seconds`) and comes back with their back to the camera alerts as "Person (face not visible)".
- If someone masked arrives at the moment a trusted person who was already in the room shows their face for the first time in 10s, that face can account for the arrival and suppress the alert.

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

The face models run on `onnxruntime-gpu`, which reuses torch's CUDA libraries. `camwatch doctor` shows whether they run `on cuda` or `on cpu`. If onnxruntime can't use the GPU, they fall back to the CPU by themselves; if that makes the dashboard's inference FPS drop, lower `face.check_fps` (full-frame face scans per second per camera).

**Upgrading from YuNet + SFace:** on first start the face database is re-embedded from the photos in `data/faces/` and `data/unknown/` (this takes a moment and is logged). Anyone with no usable photo has to be enrolled again. `face.detector_score` and `face.match_threshold` are reset to the new models' defaults.

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
- **Photo folders:** each person's photos live in `data/faces/<id>/` (the id is shown under People & faces). To fix a mix-up or add photos by hand, move/copy/delete photos there, then run Menu → Settings → *Rebuild face database from the photos in the faces folder*. Recognition uses the database, not the photos, so changes only take effect after a rebuild. It shows a before/after preview and changes nothing until you confirm.

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
| Alerts for people on the street far away | Raise `detection.person_min_height` (e.g. 0.25) |
| Wrong name assigned | Move the wrong photos to the right person's folder (or delete them), then Settings → *Rebuild face database*. Raise `face.match_threshold` (0.4–0.45) and enroll more varied samples |
| Known person shown as unknown | Enroll more samples (different light and angles), or lower `match_threshold` slightly (not below 0.28) |
| Faces often drawn as "unclear face" | Lower `face.quality_min_sharpness` or raise `face.quality_max_turn` (more faces vote, but more wrong votes) |
| Trusted person arriving gets "Person (face not visible)" | Their face isn't seen clearly twice during the 5s clip. Lower `face.min_face_px`, raise `face.detect_size` or `events.post_seconds`, or mount the camera at face height |
| Brief false detections become "Person (face not visible)" alerts | Raise `events.arrive_seconds` (e.g. 3) or `detection.confidence` |
| Same person alerts again after stepping out | Raise the camera's **Leave time** (`leave_seconds`); for recognised faces also `events.cooldown_seconds` |
| Passers-by outdoors are merged into one alert | Lower that camera's `leave_seconds` (e.g. 5–10) |
| Too many Telegram alerts | Raise `telegram.min_alert_interval_seconds` |
| Faraway faces aren't recognised | Raise `face.detect_size` to 960 or 1280 (costs more) |
| GPU overloaded | `detection.model: yolo11n.pt`, or lower `detection.detect_fps` / `face.check_fps` |

All settings are documented in `config.example.yaml`.

## Development

```bash
pip install -e .[dev]
pytest
```

## Licenses

camwatch uses Ultralytics YOLO (AGPL-3.0), so this project is AGPL-3.0 as well. The face models are InsightFace's buffalo_l pack (SCRFD + ArcFace): the code is MIT, but the pretrained weights are for **non-commercial use only**.
