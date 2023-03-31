which_install_soft() {
    answer=-1
    WHICH_CMD=$(which apt)
    if [[ ! -z $WHICH_CMD ]]; then
    	answer=0
    fi
    WHICH_CMD=$(which dnf)
    if [[ ! -z $WHICH_CMD ]]; then
    	answer=1
    fi
    WHICH_CMD=$(which pacman)
    if [[ ! -z $WHICH_CMD ]]; then
    	answer=2
    fi
}

install_pck() {
    status="$(dpkg-query -W --showformat='${db:Status-Status}' "$pkg" 2>&1)"
    if [ ! $? = 0 ] || [ ! "$status" = installed ]; then
        which_install_soft
        echo $answer
        case $answer in
            0)
                sudo apt install $pkg
                ;;
            1)
                sudo dnf install $pkg
                ;;
            2)
                sudo pacman -S $pkg
                ;;
        esac
    fi
}

pkg=texlive-full
install_pck

pip3 install matplotlib

python3 main.py
