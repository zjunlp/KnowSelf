#!/bin/sh
while test $# -gt 0
do
  case $1 in
      --prefix | -p )
          shift
          PREFIX=$1
	  shift
	  ;;
      * )
          echo "Usage: $0 [ --prefix | -p prefix-directory ]"
	  echo "  (default is /usr/local)"
          exit 1
	  ;;
  esac
done

if [ -z "$PREFIX" ]; then
    PREFIX=/usr/local
fi

if [ -d "$PREFIX/share/inform7" ]; then
    echo -n "Uninstalling inform7 from $PREFIX..."
    rm -rf "$PREFIX/share/inform7"
    rm "$PREFIX/bin/i7"
    rm "$PREFIX/man/man1/i7.1"
    mp=$(ls "$PREFIX/man/man1")
    if [ -z "$mp" ]; then
	rmdir "$PREFIX/man/man1"
    fi
    echo "done."
else
    echo "Could not find an inform7 installation under $PREFIX."
    exit 1
fi
exit 0

