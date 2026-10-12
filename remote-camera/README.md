# Camera on another PC

Use a USB camera plugged into a cheaper PC (no graphics card needed, Linux or Windows) and run camwatch on the main computer.

The small PC runs two free programs, and nothing from camwatch:
- **ffmpeg** reads the camera and compresses the video (H.264).
- **MediaMTX** serves it on the network as an RTSP stream, with a password.

camwatch then adds it like any network camera: `rtsp://camwatch:<password>@<small-pc>:8554/cam1`.

MediaMTX starts ffmpeg by itself. If the camera is unplugged or ffmpeg stops, it tries again every 5 seconds. camwatch also reconnects on its own, so the stream comes back without anyone touching either PC.

**CPU cost on the small PC:** 1080p at 15 fps takes roughly a third to one core of a normal desktop CPU. 4K takes a few cores, or almost nothing if you use the graphics chip built into the processor (see [Tuning](#tuning)).

## Linux

1. Install ffmpeg, then MediaMTX from its [releases page](https://github.com/bluenviron/mediamtx/releases). Pick the `linux_amd64` file, or `linux_arm64` on a Raspberry Pi.
   ```bash
   sudo apt install ffmpeg
   tar xzf mediamtx_v*_linux_amd64.tar.gz mediamtx
   sudo mv mediamtx /usr/local/bin/
   ```
2. Find the camera's name. Use the one ending in `-video-index0`:
   ```bash
   ls /dev/v4l/by-id/
   ```
3. Copy the config and edit it:
   ```bash
   sudo mkdir -p /etc/mediamtx
   sudo cp mediamtx-linux.yml /etc/mediamtx/mediamtx.yml
   sudo nano /etc/mediamtx/mediamtx.yml
   ```
   - Put the `/dev/v4l/by-id/...` path after `-i`.
   - Change `pass: change-me`.
4. Start it now and at every boot:
   ```bash
   sudo cp mediamtx.service /etc/systemd/system/
   sudo systemctl enable --now mediamtx
   journalctl -u mediamtx -f      # watch it start; Ctrl+C to stop watching
   ```
   The log should show `is publishing to path 'cam1'`.

If a firewall (ufw) is on, also run `sudo ufw allow 8554/tcp`.

## Windows

1. Install ffmpeg from PowerShell with `winget install Gyan.FFmpeg`, then open a new PowerShell window so it's on the PATH.
2. Download MediaMTX from its [releases page](https://github.com/bluenviron/mediamtx/releases) (the `windows_amd64.zip` file). Unzip `mediamtx.exe` into `C:\mediamtx`.
3. Find the camera's name:
   ```powershell
   ffmpeg -hide_banner -list_devices true -f dshow -i dummy
   ```
4. Copy `mediamtx-windows.yml` to `C:\mediamtx\mediamtx.yml` and edit it:
   - Put the camera name into `-i "video=..."`, keeping the quotes.
   - Change `pass: change-me`.
5. Test it by running `C:\mediamtx\mediamtx.exe C:\mediamtx\mediamtx.yml`.
   - Allow it through the firewall when Windows asks (private networks).
   - The window should show `is publishing to path 'cam1'`.
6. Start it at log on: press Win+R, type `shell:startup`, and in that folder create a shortcut with
   - Target: `C:\mediamtx\mediamtx.exe C:\mediamtx\mediamtx.yml`
   - Start in: `C:\mediamtx`
   - Run: Minimized

For a PC that should come back on its own after a power cut, set Windows to sign in automatically (`netplwiz`) and the BIOS to power on after power loss.

In Settings → Privacy → Camera, "Let desktop apps access your camera" must be on.

Running MediaMTX as a Windows service (NSSM) also works, but some webcam drivers refuse to work from a service. That's why the startup shortcut is suggested instead.

## Check it, then add it to camwatch

1. From the camwatch computer, open `rtsp://camwatch:<password>@<small-pc>:8554/cam1` in VLC (Media → Open Network Stream). You should see the camera.
2. In camwatch: Menu → Cameras → **Add network camera (RTSP/HTTP) or video file**, using the same address.
3. `camwatch doctor` should show the camera at the resolution you set (e.g. 1920x1080).

`<small-pc>` is the small PC's IP address, or its name on the network. Give it a fixed address in your router so it doesn't change.

## More than one camera

Copy the `cam1:` block in the config as `cam2:` with the other camera's path or name. camwatch reads it as `.../cam2`.

## Tuning

All of these settings are on the ffmpeg line in the config.

| Setting | Default | Why |
|---|---|---|
| `-video_size` | `1920x1080` | Use a size the camera supports (`camwatch doctor` on a PC with the camera lists them). For 4K use `3840x2160`. |
| `-input_format mjpeg` / `-vcodec mjpeg` | on | Most webcams only send 1080p and 4K in MJPEG. Without it they quietly fall back to 640x480. |
| `-framerate` | `15` | camwatch only checks a few frames per second; 15 keeps clips smooth and halves the work compared with 30. |
| `-b:v` / `-maxrate` | `6M` | The video bitrate. Too low and faces get smeared, which hurts face recognition. Use 6M for 1080p and 16M for 4K. |
| `-g` | `30` | A full picture every 2 seconds at 15 fps, so camwatch connects quickly and clips start clean. |
| H.264 on the graphics chip | off | Uses the graphics built into Intel/AMD processors instead of the CPU, worth it for 4K. The config has the line to swap in for each OS: `h264_vaapi` on Linux; `h264_qsv` (Intel) or `h264_amf` (AMD) on Windows. If ffmpeg then fails to start, go back to `libx264`. |

## If it doesn't work

- **`journalctl -u mediamtx` or the MediaMTX window repeats `runOnInit command exited`:** ffmpeg can't open the camera.
  1. Stop MediaMTX.
  2. Run the ffmpeg command from the config by hand, with `-loglevel error` changed to `-loglevel info` and everything from `-c:v` onwards replaced by `-f null -`.
  3. The error says what's wrong. Usually it's the camera path or name, or a size or frame rate the camera doesn't offer.
- **camwatch says it can't open the stream:** check that VLC on the camwatch computer can open it, that the password matches, and that port 8554 isn't blocked by a firewall.
- **The Linux service fails with "supplementary groups":** that system has no `render` group. Remove `render` from `SupplementaryGroups` in `/etc/systemd/system/mediamtx.service`, then run `sudo systemctl daemon-reload && sudo systemctl restart mediamtx`.
- **Some MediaMTX versions reject the config (`unknown field "moq"`):** versions before 1.19 don't know `moq`. Delete that line.
