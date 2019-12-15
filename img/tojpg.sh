path="/home/tetsuya/code/cpp/img/"
for a in `ls $path | grep -E "\.ppm|\.pgm"`
	do
		a=$path$a
		echo $a
		ffmpeg -i $a `echo $a | grep -E "^[^\.]+" -o`.jpg -y
	done
