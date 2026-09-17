# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# THIS IS AUTOMATICALLY INSERTED INTO $ZSH/tools/upgrade.sh
printf "\n${BLUE}%s${RESET}\n" "Updating custom plugins and themes"
cd custom/ || exit
for plugin in plugins/*/ themes/*/; do
  if [ -d "$plugin/.git" ]; then
    printf "${YELLOW}%s${RESET}\n" "${plugin%/}"
    git -C "$plugin" pull
  fi
done
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
