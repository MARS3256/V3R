# V3R

V3R is a Blender 5.2 add-on for reconstructing drone footage as a 3D map. It provides a dedicated sidebar panel for selecting aerial video, configuring reconstruction parameters, and starting or stopping the reconstruction workflow.

> **Project status:** The Blender add-on interface and reconstruction controls are available. The underlying real-time reconstruction pipeline is currently being developed.

## Features

- Blender 5.2 add-on integration
- Video file selection for common formats:
  - MP4
  - AVI
  - MKV
  - MOV
  - WebM
  - M4V
  - MPG/MPEG
- Live stream URL input
- Start and stop reconstruction controls
- Configurable camera field of view
- Configurable short-term memory frame count
- Optional debug mode

## Installation

1. Download or clone this repository.
2. In Blender, open **Edit > Preferences > Add-ons**.
3. Select **Install...** and choose the add-on package or the repository folder containing `V3R`.
4. Enable the **V3R** add-on.

The add-on requires **Blender 5.2.0 or later**.

## Usage

1. Open a 3D Viewport.
2. Press `N` to open the sidebar.
3. Select the **V3R** tab.
4. Choose a drone video with **Open File**, or provide a live **Stream Link**.
5. Expand **Settings** if you need to change:
   - **Debug Mode** for additional diagnostics
   - **Camera FOV** for the source camera's field of view
   - **STM Frames** for the number of frames used as short-term spatial memory
6. Click **Start** to begin reconstruction and **Stop** to end it.

## Project structure

```text
V3R/
├── blender_manifest.toml
├── __init__.py
├── operators/
│   ├── reconstruction.py
│   └── select_video.py
├── panels/
│   └── v3r_panel.py
├── properties/
│   └── v3r_settings.py
└── services/
    └── reconstruction_service.py
```

## Development

The add-on is written in Python using Blender's `bpy` API. The reconstruction service is designed to run asynchronously so that Blender remains responsive while processing video or live streams.

## License

V3R is distributed under the **GNU General Public License v3.0 or later (GPL-3.0-or-later)**.

## Authors

- Muhammad As-Adur Rahman Sajid
- Abu Bakor Siddik
- Priyangbada Pikee Ranjinee
- Shahriar Tanveer
- Zeesad E Arham
