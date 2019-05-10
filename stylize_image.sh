set -e
# Get a carriage return into `cr`
cr=`echo $'\n.'`
cr=${cr%.}

if [ "$#" -le 1 ]; then
   echo "Usage: bash stylize_image.sh <path_to_content_image> <path_to_style_image>"
   exit 1
fi

# Parse arguments
content_image="$1"
content_filename=$(basename "$content_image")

style_image="$2"
style_filename=$(basename "$style_image")

echo "Rendering stylized image. This may take a while..."
python neural_style.py \
--content_img "${content_filename}" \
--style_imgs "${style_filename}" \
--device "/gpu:0";
