#!/bin/sh

VERSION="6M62"

OS=$(uname)

if [ "$OS" = "SunOS" ]; then
    ARCH=$(uname -p)
    OS="$OS-$(uname -r)"
else
    ARCH=$(uname -m)
fi

case $ARCH in
    i?86 )
        ARCH=i386
    ;;
esac

case $ARCH in
    armv5* )
        ARCH=armv5tel
    ;;
    armv6* | armv7*)
	# Assuming we're building on a Pi.
	if [ -d "/lib/arm-linux-gnueabihf" ]; then
            ARCH=armv6lhf
	else
	    ARCH=armv6l
	fi
    ;;
esac

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

COMMON="inform7-common_${VERSION}_all.tar.gz"
COMPDEP="inform7-compilers_${VERSION}_${ARCH}.tar.gz"
TERPDEP="inform7-interpreters_${VERSION}_${ARCH}.tar.gz"

if [ ! -f ${COMMON} ]; then
    echo "File ${COMMON} not present!  Giving up!"
    exit 2
fi


if [ ! -f ${COMPDEP} ]; then
    echo "You should have the executable compilers for the ${ARCH} architecture."
    case $ARCH in
	x86_64 | amd64 )
            OLDARCH=${ARCH}
	    ARCH="i386"
	    COMPDEP="inform7-compilers_${VERSION}_${ARCH}.tar.gz"
	    TERPDEP="inform7-interpreters_${VERSION}_${ARCH}.tar.gz"
	    echo "${OLDARCH} binaries not found."
	    echo "Changing architecture to ${ARCH} and retrying."
	    if [ ! -f ${COMPDEP} ]; then
		echo "You do not have the executable compilers for the ${ARCH} architecture either."
		echo "Giving up!"
		exit 2
	    fi
	    ;;
	* )
	    ;;
    esac
    echo "Giving up!"
    exit 2
fi
if [ ! -f ${TERPDEP} ]; then
    echo "File ${TERPDEP} not present!"
    echo "You should have the executable interpreters for the ${ARCH} architecture."
    echo "Giving up!"
    exit 2
fi

D=$(pwd)

mkdir -p "$PREFIX"
cd "$PREFIX" || exit 3
tar xzf ${D}/${COMMON} || exit 4
tar xzf ${D}/${COMPDEP} || exit 4
tar xzf ${D}/${TERPDEP} || exit 4

if [ "$PREFIX" != "/usr/local" ]; then
    cmd="s|/usr/local|$PREFIX|;"
    perl -p -i -e "$cmd" bin/i7 || exit 6
fi

cd ${D}
exit 0

