mkdir -p images/val
mkdir -p images/train

for f in labels/val/*.txt; do
  b=$(basename "$f" .txt)
  mv images/"$b".png images/val/
done

for f in labels/train_yolo/*.txt; do
  b=$(basename "$f" .txt)
  mv images/"$b".png images/train/
done
