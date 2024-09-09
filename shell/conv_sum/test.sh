#!/bin/bash

showHelp() {
	cat <<EOF
Usage: train magiclm nano
-h, -help,          --help                  Display help
-e, -espo-version,  --epoch					num training epoch
-l, -rebuild,       --lr					learning rate

EOF
}

# $@ is all command line parameters passed to the script.
# -o is for short options like -v
# -l is for long options with double dash like --version
# the comma separates different long options
# -a is for long options with single dash like -version
options=$(getopt -l "help,epoch:,lr:,dataset:" -o "e:l:d:" -a -- "$@")

# set --:
# If no arguments follow this option, then the positional parameters are unset. Otherwise, the positional parameters
# are set to the arguments, even if some of them begin with a ‘-’.
eval set -- "$options"

echo $options

while true; do
	case "$1" in
	-h | --help)
		showHelp
		exit 0
		;;
	-e | --epoch)
		shift
		epoch="$1"
		;;
	-l | --lr)
		shift
		lr="$1"
		;;
	-d | --dataset)
		shift
		dataset=$1
		;;
	--)
		shift
		break
		;;
	esac
	shift
done

echo ${epoch}
echo ${lr}
echo ${dataset}
