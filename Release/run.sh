#/bin/sh
echo "Program start..."
Size="1280x720"
FPS="15"
PROG="./gl_texture_mapping_try"
OUTPUT=$2

if [$1 == 1];then #using ffmpeg instead of ffplay
	$PROG | ffmpeg -y -re -f rawvideo -vcodec rawvideo -pixel_format bgr24 -video_size $Size -framerate $FPS -i - -c:v libx264 $OUTPUT
elif [ $1 == 2 ];then
	$PROG | ffplay -f rawvideo -vcodec rawvideo -pixel_format bgr24 -video_size $Size -framerate $FPS -i -
else 
	$PROG | ffmpeg -re -f rawvideo -vcodec rawvideo -pixel_format bgr24 -video_size $Size -framerate $FPS -i - -deinterlace -vcodec libx264 -pixel_format yuv420p -preset medium -r 30 -g $((30 * 2)) -b:v 2500k -f flv "$OUTPUT"
		
fi
