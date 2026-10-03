# Object detection and tracking (runs on Vercel)

Real-time object detection and tracking that runs entirely in the browser.
Vercel only serves static files, so there is no server, no build step and no Python runtime to configure.

| Task requirement | Implementation |
|---|---|
| Real-time video input | Webcam (`getUserMedia`) or a local video file |
| Pre-trained model | COCO-SSD via TensorFlow.js (80 object classes) |
| Process each frame | Every frame is drawn to a canvas and sent to the detector |
| Object tracking | SORT: Kalman filter + Hungarian matching (`sort.js`) |
| Output with labels and IDs | Boxes, `class #ID` labels, trails and a live list |

## Deploy

**Option A: Vercel CLI**
```
npm i -g vercel
cd object-tracking-web
vercel --prod
```
When asked, accept the defaults (Framework Preset: Other, no build command, output directory `.`).

**Option B: GitHub**
1. Push this folder to a GitHub repository.
2. In Vercel choose **Add New > Project** and import the repository.
3. Framework Preset: **Other**. Leave build command and output directory empty. Deploy.

## Run locally
```
npx serve .
```
Open the printed address. Browsers allow the webcam on `localhost` and on `https` pages (Vercel is https).

## Files
- `index.html`: page structure and script loading
- `style.css`: styling
- `app.js`: video input, detection loop, drawing
- `sort.js`: SORT tracker (tested with synthetic crossing objects)
- `vercel.json`: camera permission header

## Notes
- The model and TensorFlow.js load from public CDNs the first time the page opens.
- Speed depends on the device. The "Fast" model is the default; "Accurate" is slower but finds smaller objects.
- To use YOLOv8 instead, export `yolov8n.onnx`, place it in this folder and run it with `onnxruntime-web`. Only the detection call in `app.js` needs to change; `sort.js` works with any detector that returns boxes.
