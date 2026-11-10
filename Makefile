# DreamCamera DSi — BlocksDS build.
# DSi mode: the ARM9 linker specs MUST be dsi_arm9.specs (DS layout is
# ds_arm9.specs). The workflow also passes SPECS on the make command line,
# which overrides anything in the included rules file.
NAME          := dreamcamera
GAME_TITLE    := DreamCamera
GAME_SUBTITLE := Deepdream camera
GAME_AUTHOR   := Coby
GAME_ICON     := dsi/icon.bmp
SOURCEDIRS    := dsi/source
SPECS         := $(BLOCKSDS)/sys/crts/dsi_arm9.specs

RULES := $(BLOCKSDS)/sys/default_makefiles/rom_arm9/Makefile
include $(RULES)
