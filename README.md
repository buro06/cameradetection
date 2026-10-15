# camwatch

24/7 multi-camera person and face detection with Telegram and/or ntfy alerts.

- **Cameras:** USB webcams, RTSP, HTTP(S) MJPEG/HLS streams, JPEG snapshot URLs, and video files (for testing). Any number of cameras can run at once.
- **Person detection:** Ultralytics YOLO11, batched across all cameras on the GPU.
- **Face recognition:** InsightFace **buffalo_l**: **SCRFD-10G** (detection) + **ArcFace R50** (recognition), run through onnxruntime (GPU if it works, otherwise CPU). The model weights are free for non-commercial use only, e.g. at home.
- **Telegram alerts** include a 5-second H.264 clip, starting the moment the person is detected. The caption names everyone seen (🟢 trusted, 🟠 known, 🔴 unknown).
- **ntfy alerts** (ntfy.sh or your own server) push the same text with the clip attached.
- **Trusted people** don't trigger alerts. Unknown faces are saved so you can label them later from Telegram or the CLI.
- **Hard cooldown:** each camera sends at most one alert (on any channel) per 30s (`telegram.min_alert_interval_seconds`); later ones are still clipped and logged as `rate-limited`.
- **Interactive dashboard** with menus. A headless mode is available for running as a service.

## How alerts are decided

Three rules per camera, all on one time span, `events.away_seconds` (30s): **someone has to be gone 30s before coming back alerts again.**

1. **People (YOLO).** When someone appears in an empty camera, that's an **arrival**. Empty means no person seen for 30s, and the person must be in view for `events.arrive_seconds` (2s) to count. A box must be at least `detection.person_min_height` (15%) of the frame height, or twice that if cut off at the frame edge, so an arm poking into view doesn't count.
2. **Faces (InsightFace), looked for only inside person boxes.**
   - Each clear face is identified: a known person, or grouped with similar unknown faces.
   - A face is **new** if this camera hasn't seen it for 30s and hasn't seen it at all since the camera was last empty. So someone sitting on the couch facing away doesn't re-alert when they turn round.
   - Blurry or turned-away faces are ignored (`face.quality_min_sharpness`, `face.quality_max_turn`).
3. **An arrival or a new face starts an event:**
   - a 5s clip (`events.post_seconds`), starting when the person first appeared, while faces keep being identified
   - faces seen clearly at least `face.min_sightings` times (2) are the people in it
   - if someone arrived and no face was clear enough (mask, hood, dark room, back turned), it's a **"Person (face not visible)"**
   - no alert if every new person is trusted

An alert still isn't *sent* if the same camera sent one less than `telegram.min_alert_interval_seconds` (30s) ago. It's clipped and logged as `rate-limited`.

**Known limits:**
- A household member who's been gone 30s and comes back with their back to the camera alerts as "Person (face not visible)".
- Someone masked who joins a person already in view isn't reported, because only an arrival in an empty camera is noticed without a face.

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

The face models run on `onnxruntime-gpu`, which reuses torch's CUDA libraries. Version 1.27 and later are built for CUDA 13 only, which doesn't support the M4000, so camwatch requires a version below 1.27. An install made before that limit was added can be fixed with `pip install "onnxruntime-gpu<1.27"`. `camwatch doctor` points this out when it happens. `camwatch doctor` shows whether they run `on cuda` or `on cpu`. If onnxruntime can't use the GPU, they fall back to the CPU by themselves; if that makes the dashboard's inference FPS drop, lower `face.check_fps` (face scans per second per camera while someone is in view).

**Upgrading from YuNet + SFace:** on first start the face database is re-embedded from the photos in `data/faces/` and `data/unknown/` (this takes a moment and is logged). Anyone with no usable photo has to be enrolled again. `face.detector_score` and `face.match_threshold` are reset to the new models' defaults.

On macOS or Linux: `python -m venv .venv && . .venv/bin/activate && pip install -e .` (CPU, CUDA, or Apple MPS are picked automatically).

## First run

```powershell
camwatch setup      # Telegram bot, ntfy + cameras wizard (writes config.yaml)
camwatch            # start monitoring with the live dashboard
```

**Telegram:**
1. Create a bot with **@BotFather** (`/newbot`).
2. Paste the token into the setup wizard.
3. Send `/start` to your bot, or add it to a group.
4. Choose **Find chats & users**.

Alerts go to `chat_ids`. Commands are accepted only from those chats or from `allowed_user_ids`; everything else is ignored and logged. To keep the token out of `config.yaml`, set it in the `CAMWATCH_TELEGRAM_TOKEN` environment variable instead.

**ntfy:** Menu → **ntfy** (or the setup wizard) asks for the server URL, the topic and, if your server uses access control, a username and password. The login is checked before it's saved. Then choose **Send test message**. Subscribe to the topic in the ntfy app on your phone.

- Each alert is titled with the camera and time. The body lists everyone seen (🟢 trusted, 🟠 known, 🔴 unknown), and the clip is attached (`ntfy.attach_clip`).
- A self-hosted server only accepts the clip if `attachment-cache-dir` is set in its `server.yml`. Its default size limit is 15 MB. If the server refuses the clip, the alert is sent as text with the reason.
- `ntfy.priority` sets the notification priority, from 1 (min) to 5 (urgent). The default is 4 (high).
- To keep the password out of `config.yaml`, set it in the `CAMWATCH_NTFY_PASSWORD` environment variable. For NSSM, use `AppEnvironmentExtra`; for systemd on Linux, use `Environment=` in the unit.

### USB cameras

Menu → Cameras → **Add USB camera** lists the cameras that are plugged in by name (e.g. `Logitech BRIO — up to 3840x2160`), then the resolutions that camera supports. camwatch then tests the camera:

- It tries the capture setups for your OS (MJPG or YUY2, DirectShow or Media Foundation on Windows, V4L2 on Linux) until the frames really come out at that size, and saves the one that works.
- It shows a test picture so you can check it's the right camera.

The camera is saved by name (`source: "usb:Logitech BRIO"`), so it stays the same camera when it's re-plugged or the PC restarts. Two identical cameras are told apart as `#2`. Cameras added by index (`source: "0"`) still work; their menu has **Use this camera by name instead of index**.

`camwatch doctor` lists the USB cameras it can see, with their names and largest resolution.

### Camera on another PC

A USB camera can also be plugged into a cheaper PC (Linux or Windows, no graphics card needed) that streams it to the computer running camwatch.

That PC runs only MediaMTX and ffmpeg. They compress the video and share it as a password-protected RTSP stream, which you add here as a network camera: `rtsp://camwatch:<password>@<small-pc>:8554/cam1`.

[`remote-camera/`](remote-camera/README.md) has ready-made configs and step-by-step setup for both systems.

### Dashboard keys

| Key | Action |
|---|---|
| `m` | Menu: cameras, people & faces, review unknown faces, arm/disarm, Telegram, settings |
| `a` | Arm/disarm all cameras |
| `v` | Live view: a resizable window per running camera showing the picture and current detections (press again to close) |
| `s` | Save a snapshot of every camera to `data/snapshots/` |
| `q` | Quit |

Monitoring keeps running while menus are open. Live view windows are hidden while a menu is open and come back when you return to the dashboard. To close a single window, click its close button. Pressing Esc or `q` in any live view window closes all of them.

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
nssm set camwatch AppEnvironmentExtra CAMWATCH_TELEGRAM_TOKEN=123456:ABC... CAMWATCH_NTFY_PASSWORD=...
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
| USB webcam stuck at 640×360 / 640×480 | Menu → Cameras → *camera* → **Resolution**. It lists the sizes the camera supports, then tries the capture setups (MJPG/YUY2, DirectShow/Media Foundation or V4L2) until one really delivers that size, and saves it. The Cameras table shows `640x360 (asked 1920x1080)` when the camera ignores the request |
| USB webcam stuck at ~5 fps at 1080p | Menu → Cameras → *camera* → **Frame rate** → 30 fps (or `fps: 30` under the camera in `config.yaml`). The log line `USB camera opened: 1920x1080 MJPG @ 30 fps` shows what the camera accepted. A warning after 5 s means frames still arrive slowly. If the format shown is YUY2, set **Pixel format** → MJPG or try backend msmf; if it's MJPG, the room is probably too dim |
| Alerts for people on the street far away | Raise `detection.person_min_height` (e.g. 0.25) |
| Wrong name assigned | Move the wrong photos to the right person's folder (or delete them), then Settings → *Rebuild face database*. Raise `face.match_threshold` (0.4–0.45) and enroll more varied samples |
| Known person shown as unknown | Enroll more samples (different light and angles), or lower `match_threshold` slightly (not below 0.28) |
| Faces often drawn as "unclear face" | Lower `face.quality_min_sharpness` or raise `face.quality_max_turn` (more faces vote, but more wrong votes) |
| Trusted person arriving gets "Person (face not visible)" | Their face isn't seen clearly twice during the 5s clip. Lower `face.min_face_px`, raise `events.post_seconds`, or mount the camera at face height |
| Brief false detections become "Person (face not visible)" alerts | Raise `events.arrive_seconds` (e.g. 3) or `detection.confidence` |
| Same person alerts again after stepping out | Raise `events.away_seconds` (e.g. 60) |
| Passers-by outdoors are merged into one alert | Lower `events.away_seconds` (e.g. 10) |
| Too many alerts (Telegram or ntfy) | Raise `telegram.min_alert_interval_seconds` (applies to every channel) |
| Faraway faces aren't recognised | Lower `face.min_face_px` (e.g. 30), or move the camera closer |
| Det/s on the dashboard below `detection.detect_fps` (slow PC, no GPU) | `detection.model: yolo11n.pt`, `detection.imgsz: 480`, `detection.detect_fps: 3`, and `face.check_fps: 1`. Faces are only scanned while someone is in view, so an empty scene costs YOLO alone |

All settings are documented in `config.example.yaml`.

## Development

```bash
pip install -e .[dev]
pytest
```

## Licenses

camwatch uses Ultralytics YOLO (AGPL-3.0), so this project is AGPL-3.0 as well. The face models are InsightFace's buffalo_l pack (SCRFD + ArcFace): the code is MIT, but the pretrained weights are for **non-commercial use only**.
