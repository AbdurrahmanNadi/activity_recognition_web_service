#!/bin/bash
if [ $(pwd | rev | cut -f 1 -d '/' | rev) == 'activity_recognition_web_service' ]; then
  mkdir -p "kinetics-i3d/data/checkpoints"
  cd "kinetics-i3d" || exit
  git init || exit
  git config core.sparseCheckout true
  echo '/data/checkpoints/*' > .git/info/sparse-checkout
  git remote add origin https://github.com/deepmind/kinetics-i3d.git
  get pull origin master
  mv data/checkpoints ../models/i3d/data/checkpoints || exit
  cd ../ && rm -rf "kinetics-i3d"
  echo 'Pretrained Model Checkpoints downloaded'
  echo '------------------------------------------'
else
    echo 'You can only run this file inside the repo'
fi