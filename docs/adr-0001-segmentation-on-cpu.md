# ADR 0001: LR-ASPP MobileNetV3 exported to ONNX for condition state segmentation

## Context

Bridge owners record steel corrosion as element condition states: the share of an element in
CS1 good, CS2 fair, CS3 poor and CS4 severe. A photo classifier that says "corrosion yes or no"
does not produce that. A per pixel map does, because the share of pixels in each state is the
same kind of quantity the inspector writes down.

The only public dataset annotated with those four states is the Virginia Tech Corrosion
Condition State dataset (440 photos from VDOT inspection reports, CC0). Its authors report an
F1 of 86.67 with DeepLabV3+ on a GPU. Recent work goes further: a two level SegFormer and
Mask2Former framework from Saitama University (published online August 2026) separates
corrosion patterns, and its authors note it has not been tested on edge hardware.

The constraint here is the opposite end: the model has to run on a laptop CPU or a field
tablet, next to the photos, with no GPU and no internet.

## Decision

- Fine-tune torchvision's LR-ASPP MobileNetV3-Large, starting from its COCO weights, on the
  396 training photos at 320 x 320, with class weights from the square root of the inverse
  pixel frequency. The rare class (CS4) is under 3% of pixels.
- Select the epoch on a 10% validation slice of the training split. The 44 test photos are
  used once, at the end.
- Export to ONNX (opset 17) and run it with ONNX Runtime on CPU. Torch is only needed to
  train, it is an optional extra.
- Turn the mask into quantities in plain code: percent of the photo per state, the worst
  state above a 0.5% noise floor, and the mean pixel confidence.

## Alternatives considered

- DeepLabV3+ or SegFormer: better masks, but several times the compute per photo and the
  point of this project is the decision layer on a CPU budget.
- A photo level classifier: cheaper, but cannot give area shares and hides where the model
  is looking.
- A vision language model that "describes the corrosion": no calibrated quantities, slow on
  CPU, and hard to audit.

## Consequences

- The masks are weak on CS4 (IoU 0.13 on the 44 test photos, against 0.44 for CS2 and 0.40
  for CS3). That is measured and reported, not hidden. The workflow is built so this error is contained: CS3 and CS4
  always go to an engineer, and low confidence sends any photo to review.
- The percentages are of the photo, not of the element. Turning them into element quantities
  needs the element area in the frame, which a single photo does not give.
- Inference is 13.6 ms per photo (median) with two CPU threads, so the model is never the
  bottleneck of the workflow.
- Swapping in a stronger model only means exporting a new ONNX file with the same input and
  output names.
