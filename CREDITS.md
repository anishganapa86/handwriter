# Credits

This project stands on other people's work.

| What | Who | Licence | Where it is used |
|---|---|---|---|
| Head mount and spring modules for the P1S (Universal Modular Tool System) | Falu Mendy Vergara, [MakerWorld model 2029113](https://makerworld.com/en/models/2029113-modular-system-for-a1-p1-x1-series) | His MakerWorld licence: personal use; commercial use needs his membership | `hardware/falu/` (his files, unchanged); the outside of `hardware/out/pencil_guide.stl` copies his tool guide; Plotter tab preset uses the clearances of his P1S profile |
| P1S setup guide for Falu's system | [rebelthor/bambu-lab-pen-plotter](https://github.com/rebelthor/bambu-lab-pen-plotter) | MIT | Procedure and collision warnings behind the Plotter tab steps |
| P1S pen plotting approach and Bambu Studio profile | [gtls64/p1s-pen-plotter](https://github.com/gtls64/p1s-pen-plotter) | MIT | `ml/hw/plotter.py`, `ml/weights/bambu/` |
| Handwriting synthesis network and English weights | [X-rayLaser/pytorch-handwriting-synthesis-toolkit](https://github.com/X-rayLaser/pytorch-handwriting-synthesis-toolkit) | MIT | `ml/hw/model.py`, `ml/weights/base/` |
| InkSight photo-to-ink model | Google Research | Apache-2.0 | `ml/inksight_worker.py` (downloaded at run time) |
| TrOCR, EasyOCR | Microsoft, JaidedAI | MIT, Apache-2.0 | Reading text in photos (downloaded at run time) |

Falu's files must not be redistributed. Keep this repository private while they are in it, or delete `hardware/falu/` before sharing it.
