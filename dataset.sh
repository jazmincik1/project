#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <path-to-dataset-directory>"
  exit 1
fi

DATASET_DIR="$1"

# translate = {
#  "cane": "dog",
#  "cavallo": "horse",
#  "elefante": "elephant",
#  "farfalla": "butterfly",
#  "gallina": "chicken",
#  "gatto": "cat",
#  "mucca": "cow",
#  "pecora": "sheep",
#  "scoiattolo": "squirrel",
#  "dog": "cane",
#  "cavallo": "horse",
#  "elephant" : "elefante",
#  "butterfly": "farfalla",
#  "chicken": "gallina",
#  "cat": "gatto",
#  "cow": "mucca",
#  "spider": "ragno",
#  "squirrel": "scoiattolo"}

declare -A translate=(
    ["cane"]="dog"
    ["cavallo"]="horse"
    ["elefante"]="elephant"
    ["farfalla"]="butterfly"
    ["gallina"]="chicken"
    ["gatto"]="cat"
    ["mucca"]="cow"
    ["pecora"]="sheep"
    ["scoiattolo"]="squirrel"
    ["ragno"]="spider"
)

cd "$DATASET_DIR" || exit

for folder in */ ; do
  folder_name=${folder%/}
  
  if [[ -n "${translate[$folder_name]}" ]]; then
    mv "$folder_name" "${translate[$folder_name]}"
    echo "Renamed $folder_name to ${translate[$folder_name]}"
  else
    echo "No mapping found for $folder_name, skipping..."
  fi
done

