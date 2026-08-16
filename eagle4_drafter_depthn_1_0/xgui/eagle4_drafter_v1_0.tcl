# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "ACCFIX_B2P_P" -parent ${Page_0}
  ipgui::add_param $IPINST -name "ACCFIX_KC_P" -parent ${Page_0}
  ipgui::add_param $IPINST -name "ACCFIX_KVONLY_P" -parent ${Page_0}
  ipgui::add_param $IPINST -name "ACCFIX_SEAM_P" -parent ${Page_0}
  ipgui::add_param $IPINST -name "ARR_GROUPS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "ARR_PIPE_STAGES" -parent ${Page_0}
  ipgui::add_param $IPINST -name "A_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BACK_SEQ" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BODY_BATCH_B" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BOOT_AXI_ADDR_W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BOOT_AXI_DATA_W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BOOT_AXI_ID_W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BOOT_BURST_LEN" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BUFRDREG" -parent ${Page_0}
  ipgui::add_param $IPINST -name "COMBINE_LOCAL" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CONNECT_SEAMS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CTX_WIN" -parent ${Page_0}
  ipgui::add_param $IPINST -name "D1_LAT" -parent ${Page_0}
  ipgui::add_param $IPINST -name "D1_RDLAT" -parent ${Page_0}
  ipgui::add_param $IPINST -name "D1_WLAT" -parent ${Page_0}
  ipgui::add_param $IPINST -name "DEPTH_N" -parent ${Page_0}
  ipgui::add_param $IPINST -name "DUAL_PORT" -parent ${Page_0}
  ipgui::add_param $IPINST -name "EMBG_EN" -parent ${Page_0}
  ipgui::add_param $IPINST -name "EMB_NCH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "ENORM_D0" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FF_ACT_CACHE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FORREASON" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FR_ABANK_BRAM" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FR_KF_BANKS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_CONSUME_HARDEN" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_DATA_W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_K_C" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_LANE_FIX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_MAX_BURST" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_MIN_NG" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_MROUND" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_NCH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_OPT1C" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_OPT1D" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_OPT1F" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_OPT1G" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_OUTSTANDING" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_PIPELINED" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_PREFETCH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_ROWPP" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_SCALAR_II1" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_SELPIPE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_SPLICE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_WIDE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_WIDE_IL" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_WIDE_M" -parent ${Page_0}
  ipgui::add_param $IPINST -name "G8_WIDE_P" -parent ${Page_0}
  ipgui::add_param $IPINST -name "GEMV_KPAR64" -parent ${Page_0}
  ipgui::add_param $IPINST -name "GEMV_SCALED" -parent ${Page_0}
  ipgui::add_param $IPINST -name "GEMV_SCALE_FRAC" -parent ${Page_0}
  ipgui::add_param $IPINST -name "GPTQ_QZERO_RAW" -parent ${Page_0}
  ipgui::add_param $IPINST -name "HEAD_PAR" -parent ${Page_0}
  ipgui::add_param $IPINST -name "KV_ADDR_REP" -parent ${Page_0}
  ipgui::add_param $IPINST -name "KV_K_RAM_STYLE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "KV_RES_CLR_REP" -parent ${Page_0}
  ipgui::add_param $IPINST -name "KV_V_RAM_STYLE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "K_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "K_PAR" -parent ${Page_0}
  ipgui::add_param $IPINST -name "LANES" -parent ${Page_0}
  ipgui::add_param $IPINST -name "LMDOWN_MEM_DIR_P" -parent ${Page_0}
  ipgui::add_param $IPINST -name "OBS_LVL_P" -parent ${Page_0}
  ipgui::add_param $IPINST -name "OUT_ABI_V24" -parent ${Page_0}
  ipgui::add_param $IPINST -name "OUT_PARENT_SENTINEL" -parent ${Page_0}
  ipgui::add_param $IPINST -name "OUT_TOP_K" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PRECISION_SAFE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PREFETCH_COL_P" -parent ${Page_0}
  ipgui::add_param $IPINST -name "R4_BANKSPLIT" -parent ${Page_0}
  ipgui::add_param $IPINST -name "REQUANT_HEADMUX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "ROPE_CS_PIPE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "ROPE_POS_REP" -parent ${Page_0}
  ipgui::add_param $IPINST -name "S7A_FRONT_SEQ" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SEAM_FF" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SEAM_FF_LAND_P" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SEAM_R1_FULLW" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SILU_RANGE_FIX_P" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SLR_SAFE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STAGE_INJECT" -parent ${Page_0}
  ipgui::add_param $IPINST -name "TAIL_FP16" -parent ${Page_0}
  ipgui::add_param $IPINST -name "TREE_P" -parent ${Page_0}
  ipgui::add_param $IPINST -name "TREE_PP" -parent ${Page_0}
  ipgui::add_param $IPINST -name "TREE_SEAM_FF" -parent ${Page_0}
  ipgui::add_param $IPINST -name "TREE_V_MAX" -parent ${Page_0}


}

proc update_PARAM_VALUE.ACCFIX_B2P_P { PARAM_VALUE.ACCFIX_B2P_P } {
	# Procedure called to update ACCFIX_B2P_P when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ACCFIX_B2P_P { PARAM_VALUE.ACCFIX_B2P_P } {
	# Procedure called to validate ACCFIX_B2P_P
	return true
}

proc update_PARAM_VALUE.ACCFIX_KC_P { PARAM_VALUE.ACCFIX_KC_P } {
	# Procedure called to update ACCFIX_KC_P when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ACCFIX_KC_P { PARAM_VALUE.ACCFIX_KC_P } {
	# Procedure called to validate ACCFIX_KC_P
	return true
}

proc update_PARAM_VALUE.ACCFIX_KVONLY_P { PARAM_VALUE.ACCFIX_KVONLY_P } {
	# Procedure called to update ACCFIX_KVONLY_P when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ACCFIX_KVONLY_P { PARAM_VALUE.ACCFIX_KVONLY_P } {
	# Procedure called to validate ACCFIX_KVONLY_P
	return true
}

proc update_PARAM_VALUE.ACCFIX_SEAM_P { PARAM_VALUE.ACCFIX_SEAM_P } {
	# Procedure called to update ACCFIX_SEAM_P when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ACCFIX_SEAM_P { PARAM_VALUE.ACCFIX_SEAM_P } {
	# Procedure called to validate ACCFIX_SEAM_P
	return true
}

proc update_PARAM_VALUE.ARR_GROUPS { PARAM_VALUE.ARR_GROUPS } {
	# Procedure called to update ARR_GROUPS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ARR_GROUPS { PARAM_VALUE.ARR_GROUPS } {
	# Procedure called to validate ARR_GROUPS
	return true
}

proc update_PARAM_VALUE.ARR_PIPE_STAGES { PARAM_VALUE.ARR_PIPE_STAGES } {
	# Procedure called to update ARR_PIPE_STAGES when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ARR_PIPE_STAGES { PARAM_VALUE.ARR_PIPE_STAGES } {
	# Procedure called to validate ARR_PIPE_STAGES
	return true
}

proc update_PARAM_VALUE.A_WIDTH { PARAM_VALUE.A_WIDTH } {
	# Procedure called to update A_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.A_WIDTH { PARAM_VALUE.A_WIDTH } {
	# Procedure called to validate A_WIDTH
	return true
}

proc update_PARAM_VALUE.BACK_SEQ { PARAM_VALUE.BACK_SEQ } {
	# Procedure called to update BACK_SEQ when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BACK_SEQ { PARAM_VALUE.BACK_SEQ } {
	# Procedure called to validate BACK_SEQ
	return true
}

proc update_PARAM_VALUE.BODY_BATCH_B { PARAM_VALUE.BODY_BATCH_B } {
	# Procedure called to update BODY_BATCH_B when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BODY_BATCH_B { PARAM_VALUE.BODY_BATCH_B } {
	# Procedure called to validate BODY_BATCH_B
	return true
}

proc update_PARAM_VALUE.BOOT_AXI_ADDR_W { PARAM_VALUE.BOOT_AXI_ADDR_W } {
	# Procedure called to update BOOT_AXI_ADDR_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BOOT_AXI_ADDR_W { PARAM_VALUE.BOOT_AXI_ADDR_W } {
	# Procedure called to validate BOOT_AXI_ADDR_W
	return true
}

proc update_PARAM_VALUE.BOOT_AXI_DATA_W { PARAM_VALUE.BOOT_AXI_DATA_W } {
	# Procedure called to update BOOT_AXI_DATA_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BOOT_AXI_DATA_W { PARAM_VALUE.BOOT_AXI_DATA_W } {
	# Procedure called to validate BOOT_AXI_DATA_W
	return true
}

proc update_PARAM_VALUE.BOOT_AXI_ID_W { PARAM_VALUE.BOOT_AXI_ID_W } {
	# Procedure called to update BOOT_AXI_ID_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BOOT_AXI_ID_W { PARAM_VALUE.BOOT_AXI_ID_W } {
	# Procedure called to validate BOOT_AXI_ID_W
	return true
}

proc update_PARAM_VALUE.BOOT_BURST_LEN { PARAM_VALUE.BOOT_BURST_LEN } {
	# Procedure called to update BOOT_BURST_LEN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BOOT_BURST_LEN { PARAM_VALUE.BOOT_BURST_LEN } {
	# Procedure called to validate BOOT_BURST_LEN
	return true
}

proc update_PARAM_VALUE.BUFRDREG { PARAM_VALUE.BUFRDREG } {
	# Procedure called to update BUFRDREG when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BUFRDREG { PARAM_VALUE.BUFRDREG } {
	# Procedure called to validate BUFRDREG
	return true
}

proc update_PARAM_VALUE.COMBINE_LOCAL { PARAM_VALUE.COMBINE_LOCAL } {
	# Procedure called to update COMBINE_LOCAL when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.COMBINE_LOCAL { PARAM_VALUE.COMBINE_LOCAL } {
	# Procedure called to validate COMBINE_LOCAL
	return true
}

proc update_PARAM_VALUE.CONNECT_SEAMS { PARAM_VALUE.CONNECT_SEAMS } {
	# Procedure called to update CONNECT_SEAMS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CONNECT_SEAMS { PARAM_VALUE.CONNECT_SEAMS } {
	# Procedure called to validate CONNECT_SEAMS
	return true
}

proc update_PARAM_VALUE.CTX_WIN { PARAM_VALUE.CTX_WIN } {
	# Procedure called to update CTX_WIN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CTX_WIN { PARAM_VALUE.CTX_WIN } {
	# Procedure called to validate CTX_WIN
	return true
}

proc update_PARAM_VALUE.D1_LAT { PARAM_VALUE.D1_LAT } {
	# Procedure called to update D1_LAT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.D1_LAT { PARAM_VALUE.D1_LAT } {
	# Procedure called to validate D1_LAT
	return true
}

proc update_PARAM_VALUE.D1_RDLAT { PARAM_VALUE.D1_RDLAT } {
	# Procedure called to update D1_RDLAT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.D1_RDLAT { PARAM_VALUE.D1_RDLAT } {
	# Procedure called to validate D1_RDLAT
	return true
}

proc update_PARAM_VALUE.D1_WLAT { PARAM_VALUE.D1_WLAT } {
	# Procedure called to update D1_WLAT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.D1_WLAT { PARAM_VALUE.D1_WLAT } {
	# Procedure called to validate D1_WLAT
	return true
}

proc update_PARAM_VALUE.DEPTH_N { PARAM_VALUE.DEPTH_N } {
	# Procedure called to update DEPTH_N when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DEPTH_N { PARAM_VALUE.DEPTH_N } {
	# Procedure called to validate DEPTH_N
	return true
}

proc update_PARAM_VALUE.DUAL_PORT { PARAM_VALUE.DUAL_PORT } {
	# Procedure called to update DUAL_PORT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DUAL_PORT { PARAM_VALUE.DUAL_PORT } {
	# Procedure called to validate DUAL_PORT
	return true
}

proc update_PARAM_VALUE.EMBG_EN { PARAM_VALUE.EMBG_EN } {
	# Procedure called to update EMBG_EN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.EMBG_EN { PARAM_VALUE.EMBG_EN } {
	# Procedure called to validate EMBG_EN
	return true
}

proc update_PARAM_VALUE.EMB_NCH { PARAM_VALUE.EMB_NCH } {
	# Procedure called to update EMB_NCH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.EMB_NCH { PARAM_VALUE.EMB_NCH } {
	# Procedure called to validate EMB_NCH
	return true
}

proc update_PARAM_VALUE.ENORM_D0 { PARAM_VALUE.ENORM_D0 } {
	# Procedure called to update ENORM_D0 when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ENORM_D0 { PARAM_VALUE.ENORM_D0 } {
	# Procedure called to validate ENORM_D0
	return true
}

proc update_PARAM_VALUE.FF_ACT_CACHE { PARAM_VALUE.FF_ACT_CACHE } {
	# Procedure called to update FF_ACT_CACHE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FF_ACT_CACHE { PARAM_VALUE.FF_ACT_CACHE } {
	# Procedure called to validate FF_ACT_CACHE
	return true
}

proc update_PARAM_VALUE.FORREASON { PARAM_VALUE.FORREASON } {
	# Procedure called to update FORREASON when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FORREASON { PARAM_VALUE.FORREASON } {
	# Procedure called to validate FORREASON
	return true
}

proc update_PARAM_VALUE.FR_ABANK_BRAM { PARAM_VALUE.FR_ABANK_BRAM } {
	# Procedure called to update FR_ABANK_BRAM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FR_ABANK_BRAM { PARAM_VALUE.FR_ABANK_BRAM } {
	# Procedure called to validate FR_ABANK_BRAM
	return true
}

proc update_PARAM_VALUE.FR_KF_BANKS { PARAM_VALUE.FR_KF_BANKS } {
	# Procedure called to update FR_KF_BANKS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FR_KF_BANKS { PARAM_VALUE.FR_KF_BANKS } {
	# Procedure called to validate FR_KF_BANKS
	return true
}

proc update_PARAM_VALUE.G8_CONSUME_HARDEN { PARAM_VALUE.G8_CONSUME_HARDEN } {
	# Procedure called to update G8_CONSUME_HARDEN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_CONSUME_HARDEN { PARAM_VALUE.G8_CONSUME_HARDEN } {
	# Procedure called to validate G8_CONSUME_HARDEN
	return true
}

proc update_PARAM_VALUE.G8_DATA_W { PARAM_VALUE.G8_DATA_W } {
	# Procedure called to update G8_DATA_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_DATA_W { PARAM_VALUE.G8_DATA_W } {
	# Procedure called to validate G8_DATA_W
	return true
}

proc update_PARAM_VALUE.G8_K_C { PARAM_VALUE.G8_K_C } {
	# Procedure called to update G8_K_C when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_K_C { PARAM_VALUE.G8_K_C } {
	# Procedure called to validate G8_K_C
	return true
}

proc update_PARAM_VALUE.G8_LANE_FIX { PARAM_VALUE.G8_LANE_FIX } {
	# Procedure called to update G8_LANE_FIX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_LANE_FIX { PARAM_VALUE.G8_LANE_FIX } {
	# Procedure called to validate G8_LANE_FIX
	return true
}

proc update_PARAM_VALUE.G8_MAX_BURST { PARAM_VALUE.G8_MAX_BURST } {
	# Procedure called to update G8_MAX_BURST when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_MAX_BURST { PARAM_VALUE.G8_MAX_BURST } {
	# Procedure called to validate G8_MAX_BURST
	return true
}

proc update_PARAM_VALUE.G8_MIN_NG { PARAM_VALUE.G8_MIN_NG } {
	# Procedure called to update G8_MIN_NG when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_MIN_NG { PARAM_VALUE.G8_MIN_NG } {
	# Procedure called to validate G8_MIN_NG
	return true
}

proc update_PARAM_VALUE.G8_MROUND { PARAM_VALUE.G8_MROUND } {
	# Procedure called to update G8_MROUND when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_MROUND { PARAM_VALUE.G8_MROUND } {
	# Procedure called to validate G8_MROUND
	return true
}

proc update_PARAM_VALUE.G8_NCH { PARAM_VALUE.G8_NCH } {
	# Procedure called to update G8_NCH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_NCH { PARAM_VALUE.G8_NCH } {
	# Procedure called to validate G8_NCH
	return true
}

proc update_PARAM_VALUE.G8_OPT1C { PARAM_VALUE.G8_OPT1C } {
	# Procedure called to update G8_OPT1C when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_OPT1C { PARAM_VALUE.G8_OPT1C } {
	# Procedure called to validate G8_OPT1C
	return true
}

proc update_PARAM_VALUE.G8_OPT1D { PARAM_VALUE.G8_OPT1D } {
	# Procedure called to update G8_OPT1D when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_OPT1D { PARAM_VALUE.G8_OPT1D } {
	# Procedure called to validate G8_OPT1D
	return true
}

proc update_PARAM_VALUE.G8_OPT1F { PARAM_VALUE.G8_OPT1F } {
	# Procedure called to update G8_OPT1F when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_OPT1F { PARAM_VALUE.G8_OPT1F } {
	# Procedure called to validate G8_OPT1F
	return true
}

proc update_PARAM_VALUE.G8_OPT1G { PARAM_VALUE.G8_OPT1G } {
	# Procedure called to update G8_OPT1G when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_OPT1G { PARAM_VALUE.G8_OPT1G } {
	# Procedure called to validate G8_OPT1G
	return true
}

proc update_PARAM_VALUE.G8_OUTSTANDING { PARAM_VALUE.G8_OUTSTANDING } {
	# Procedure called to update G8_OUTSTANDING when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_OUTSTANDING { PARAM_VALUE.G8_OUTSTANDING } {
	# Procedure called to validate G8_OUTSTANDING
	return true
}

proc update_PARAM_VALUE.G8_PIPELINED { PARAM_VALUE.G8_PIPELINED } {
	# Procedure called to update G8_PIPELINED when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_PIPELINED { PARAM_VALUE.G8_PIPELINED } {
	# Procedure called to validate G8_PIPELINED
	return true
}

proc update_PARAM_VALUE.G8_PREFETCH { PARAM_VALUE.G8_PREFETCH } {
	# Procedure called to update G8_PREFETCH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_PREFETCH { PARAM_VALUE.G8_PREFETCH } {
	# Procedure called to validate G8_PREFETCH
	return true
}

proc update_PARAM_VALUE.G8_ROWPP { PARAM_VALUE.G8_ROWPP } {
	# Procedure called to update G8_ROWPP when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_ROWPP { PARAM_VALUE.G8_ROWPP } {
	# Procedure called to validate G8_ROWPP
	return true
}

proc update_PARAM_VALUE.G8_SCALAR_II1 { PARAM_VALUE.G8_SCALAR_II1 } {
	# Procedure called to update G8_SCALAR_II1 when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_SCALAR_II1 { PARAM_VALUE.G8_SCALAR_II1 } {
	# Procedure called to validate G8_SCALAR_II1
	return true
}

proc update_PARAM_VALUE.G8_SELPIPE { PARAM_VALUE.G8_SELPIPE } {
	# Procedure called to update G8_SELPIPE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_SELPIPE { PARAM_VALUE.G8_SELPIPE } {
	# Procedure called to validate G8_SELPIPE
	return true
}

proc update_PARAM_VALUE.G8_SPLICE { PARAM_VALUE.G8_SPLICE } {
	# Procedure called to update G8_SPLICE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_SPLICE { PARAM_VALUE.G8_SPLICE } {
	# Procedure called to validate G8_SPLICE
	return true
}

proc update_PARAM_VALUE.G8_WIDE { PARAM_VALUE.G8_WIDE } {
	# Procedure called to update G8_WIDE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_WIDE { PARAM_VALUE.G8_WIDE } {
	# Procedure called to validate G8_WIDE
	return true
}

proc update_PARAM_VALUE.G8_WIDE_IL { PARAM_VALUE.G8_WIDE_IL } {
	# Procedure called to update G8_WIDE_IL when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_WIDE_IL { PARAM_VALUE.G8_WIDE_IL } {
	# Procedure called to validate G8_WIDE_IL
	return true
}

proc update_PARAM_VALUE.G8_WIDE_M { PARAM_VALUE.G8_WIDE_M } {
	# Procedure called to update G8_WIDE_M when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_WIDE_M { PARAM_VALUE.G8_WIDE_M } {
	# Procedure called to validate G8_WIDE_M
	return true
}

proc update_PARAM_VALUE.G8_WIDE_P { PARAM_VALUE.G8_WIDE_P } {
	# Procedure called to update G8_WIDE_P when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.G8_WIDE_P { PARAM_VALUE.G8_WIDE_P } {
	# Procedure called to validate G8_WIDE_P
	return true
}

proc update_PARAM_VALUE.GEMV_KPAR64 { PARAM_VALUE.GEMV_KPAR64 } {
	# Procedure called to update GEMV_KPAR64 when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.GEMV_KPAR64 { PARAM_VALUE.GEMV_KPAR64 } {
	# Procedure called to validate GEMV_KPAR64
	return true
}

proc update_PARAM_VALUE.GEMV_SCALED { PARAM_VALUE.GEMV_SCALED } {
	# Procedure called to update GEMV_SCALED when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.GEMV_SCALED { PARAM_VALUE.GEMV_SCALED } {
	# Procedure called to validate GEMV_SCALED
	return true
}

proc update_PARAM_VALUE.GEMV_SCALE_FRAC { PARAM_VALUE.GEMV_SCALE_FRAC } {
	# Procedure called to update GEMV_SCALE_FRAC when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.GEMV_SCALE_FRAC { PARAM_VALUE.GEMV_SCALE_FRAC } {
	# Procedure called to validate GEMV_SCALE_FRAC
	return true
}

proc update_PARAM_VALUE.GPTQ_QZERO_RAW { PARAM_VALUE.GPTQ_QZERO_RAW } {
	# Procedure called to update GPTQ_QZERO_RAW when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.GPTQ_QZERO_RAW { PARAM_VALUE.GPTQ_QZERO_RAW } {
	# Procedure called to validate GPTQ_QZERO_RAW
	return true
}

proc update_PARAM_VALUE.HEAD_PAR { PARAM_VALUE.HEAD_PAR } {
	# Procedure called to update HEAD_PAR when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.HEAD_PAR { PARAM_VALUE.HEAD_PAR } {
	# Procedure called to validate HEAD_PAR
	return true
}

proc update_PARAM_VALUE.KV_ADDR_REP { PARAM_VALUE.KV_ADDR_REP } {
	# Procedure called to update KV_ADDR_REP when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.KV_ADDR_REP { PARAM_VALUE.KV_ADDR_REP } {
	# Procedure called to validate KV_ADDR_REP
	return true
}

proc update_PARAM_VALUE.KV_K_RAM_STYLE { PARAM_VALUE.KV_K_RAM_STYLE } {
	# Procedure called to update KV_K_RAM_STYLE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.KV_K_RAM_STYLE { PARAM_VALUE.KV_K_RAM_STYLE } {
	# Procedure called to validate KV_K_RAM_STYLE
	return true
}

proc update_PARAM_VALUE.KV_RES_CLR_REP { PARAM_VALUE.KV_RES_CLR_REP } {
	# Procedure called to update KV_RES_CLR_REP when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.KV_RES_CLR_REP { PARAM_VALUE.KV_RES_CLR_REP } {
	# Procedure called to validate KV_RES_CLR_REP
	return true
}

proc update_PARAM_VALUE.KV_V_RAM_STYLE { PARAM_VALUE.KV_V_RAM_STYLE } {
	# Procedure called to update KV_V_RAM_STYLE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.KV_V_RAM_STYLE { PARAM_VALUE.KV_V_RAM_STYLE } {
	# Procedure called to validate KV_V_RAM_STYLE
	return true
}

proc update_PARAM_VALUE.K_MAX { PARAM_VALUE.K_MAX } {
	# Procedure called to update K_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.K_MAX { PARAM_VALUE.K_MAX } {
	# Procedure called to validate K_MAX
	return true
}

proc update_PARAM_VALUE.K_PAR { PARAM_VALUE.K_PAR } {
	# Procedure called to update K_PAR when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.K_PAR { PARAM_VALUE.K_PAR } {
	# Procedure called to validate K_PAR
	return true
}

proc update_PARAM_VALUE.LANES { PARAM_VALUE.LANES } {
	# Procedure called to update LANES when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.LANES { PARAM_VALUE.LANES } {
	# Procedure called to validate LANES
	return true
}

proc update_PARAM_VALUE.LMDOWN_MEM_DIR_P { PARAM_VALUE.LMDOWN_MEM_DIR_P } {
	# Procedure called to update LMDOWN_MEM_DIR_P when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.LMDOWN_MEM_DIR_P { PARAM_VALUE.LMDOWN_MEM_DIR_P } {
	# Procedure called to validate LMDOWN_MEM_DIR_P
	return true
}

proc update_PARAM_VALUE.OBS_LVL_P { PARAM_VALUE.OBS_LVL_P } {
	# Procedure called to update OBS_LVL_P when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.OBS_LVL_P { PARAM_VALUE.OBS_LVL_P } {
	# Procedure called to validate OBS_LVL_P
	return true
}

proc update_PARAM_VALUE.OUT_ABI_V24 { PARAM_VALUE.OUT_ABI_V24 } {
	# Procedure called to update OUT_ABI_V24 when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.OUT_ABI_V24 { PARAM_VALUE.OUT_ABI_V24 } {
	# Procedure called to validate OUT_ABI_V24
	return true
}

proc update_PARAM_VALUE.OUT_PARENT_SENTINEL { PARAM_VALUE.OUT_PARENT_SENTINEL } {
	# Procedure called to update OUT_PARENT_SENTINEL when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.OUT_PARENT_SENTINEL { PARAM_VALUE.OUT_PARENT_SENTINEL } {
	# Procedure called to validate OUT_PARENT_SENTINEL
	return true
}

proc update_PARAM_VALUE.OUT_TOP_K { PARAM_VALUE.OUT_TOP_K } {
	# Procedure called to update OUT_TOP_K when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.OUT_TOP_K { PARAM_VALUE.OUT_TOP_K } {
	# Procedure called to validate OUT_TOP_K
	return true
}

proc update_PARAM_VALUE.PRECISION_SAFE { PARAM_VALUE.PRECISION_SAFE } {
	# Procedure called to update PRECISION_SAFE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PRECISION_SAFE { PARAM_VALUE.PRECISION_SAFE } {
	# Procedure called to validate PRECISION_SAFE
	return true
}

proc update_PARAM_VALUE.PREFETCH_COL_P { PARAM_VALUE.PREFETCH_COL_P } {
	# Procedure called to update PREFETCH_COL_P when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PREFETCH_COL_P { PARAM_VALUE.PREFETCH_COL_P } {
	# Procedure called to validate PREFETCH_COL_P
	return true
}

proc update_PARAM_VALUE.R4_BANKSPLIT { PARAM_VALUE.R4_BANKSPLIT } {
	# Procedure called to update R4_BANKSPLIT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.R4_BANKSPLIT { PARAM_VALUE.R4_BANKSPLIT } {
	# Procedure called to validate R4_BANKSPLIT
	return true
}

proc update_PARAM_VALUE.REQUANT_HEADMUX { PARAM_VALUE.REQUANT_HEADMUX } {
	# Procedure called to update REQUANT_HEADMUX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.REQUANT_HEADMUX { PARAM_VALUE.REQUANT_HEADMUX } {
	# Procedure called to validate REQUANT_HEADMUX
	return true
}

proc update_PARAM_VALUE.ROPE_CS_PIPE { PARAM_VALUE.ROPE_CS_PIPE } {
	# Procedure called to update ROPE_CS_PIPE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ROPE_CS_PIPE { PARAM_VALUE.ROPE_CS_PIPE } {
	# Procedure called to validate ROPE_CS_PIPE
	return true
}

proc update_PARAM_VALUE.ROPE_POS_REP { PARAM_VALUE.ROPE_POS_REP } {
	# Procedure called to update ROPE_POS_REP when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ROPE_POS_REP { PARAM_VALUE.ROPE_POS_REP } {
	# Procedure called to validate ROPE_POS_REP
	return true
}

proc update_PARAM_VALUE.S7A_FRONT_SEQ { PARAM_VALUE.S7A_FRONT_SEQ } {
	# Procedure called to update S7A_FRONT_SEQ when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.S7A_FRONT_SEQ { PARAM_VALUE.S7A_FRONT_SEQ } {
	# Procedure called to validate S7A_FRONT_SEQ
	return true
}

proc update_PARAM_VALUE.SEAM_FF { PARAM_VALUE.SEAM_FF } {
	# Procedure called to update SEAM_FF when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SEAM_FF { PARAM_VALUE.SEAM_FF } {
	# Procedure called to validate SEAM_FF
	return true
}

proc update_PARAM_VALUE.SEAM_FF_LAND_P { PARAM_VALUE.SEAM_FF_LAND_P } {
	# Procedure called to update SEAM_FF_LAND_P when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SEAM_FF_LAND_P { PARAM_VALUE.SEAM_FF_LAND_P } {
	# Procedure called to validate SEAM_FF_LAND_P
	return true
}

proc update_PARAM_VALUE.SEAM_R1_FULLW { PARAM_VALUE.SEAM_R1_FULLW } {
	# Procedure called to update SEAM_R1_FULLW when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SEAM_R1_FULLW { PARAM_VALUE.SEAM_R1_FULLW } {
	# Procedure called to validate SEAM_R1_FULLW
	return true
}

proc update_PARAM_VALUE.SILU_RANGE_FIX_P { PARAM_VALUE.SILU_RANGE_FIX_P } {
	# Procedure called to update SILU_RANGE_FIX_P when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SILU_RANGE_FIX_P { PARAM_VALUE.SILU_RANGE_FIX_P } {
	# Procedure called to validate SILU_RANGE_FIX_P
	return true
}

proc update_PARAM_VALUE.SLR_SAFE { PARAM_VALUE.SLR_SAFE } {
	# Procedure called to update SLR_SAFE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SLR_SAFE { PARAM_VALUE.SLR_SAFE } {
	# Procedure called to validate SLR_SAFE
	return true
}

proc update_PARAM_VALUE.STAGE_INJECT { PARAM_VALUE.STAGE_INJECT } {
	# Procedure called to update STAGE_INJECT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STAGE_INJECT { PARAM_VALUE.STAGE_INJECT } {
	# Procedure called to validate STAGE_INJECT
	return true
}

proc update_PARAM_VALUE.TAIL_FP16 { PARAM_VALUE.TAIL_FP16 } {
	# Procedure called to update TAIL_FP16 when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.TAIL_FP16 { PARAM_VALUE.TAIL_FP16 } {
	# Procedure called to validate TAIL_FP16
	return true
}

proc update_PARAM_VALUE.TREE_P { PARAM_VALUE.TREE_P } {
	# Procedure called to update TREE_P when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.TREE_P { PARAM_VALUE.TREE_P } {
	# Procedure called to validate TREE_P
	return true
}

proc update_PARAM_VALUE.TREE_PP { PARAM_VALUE.TREE_PP } {
	# Procedure called to update TREE_PP when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.TREE_PP { PARAM_VALUE.TREE_PP } {
	# Procedure called to validate TREE_PP
	return true
}

proc update_PARAM_VALUE.TREE_SEAM_FF { PARAM_VALUE.TREE_SEAM_FF } {
	# Procedure called to update TREE_SEAM_FF when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.TREE_SEAM_FF { PARAM_VALUE.TREE_SEAM_FF } {
	# Procedure called to validate TREE_SEAM_FF
	return true
}

proc update_PARAM_VALUE.TREE_V_MAX { PARAM_VALUE.TREE_V_MAX } {
	# Procedure called to update TREE_V_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.TREE_V_MAX { PARAM_VALUE.TREE_V_MAX } {
	# Procedure called to validate TREE_V_MAX
	return true
}


proc update_MODELPARAM_VALUE.LANES { MODELPARAM_VALUE.LANES PARAM_VALUE.LANES } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.LANES}] ${MODELPARAM_VALUE.LANES}
}

proc update_MODELPARAM_VALUE.K_MAX { MODELPARAM_VALUE.K_MAX PARAM_VALUE.K_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.K_MAX}] ${MODELPARAM_VALUE.K_MAX}
}

proc update_MODELPARAM_VALUE.K_PAR { MODELPARAM_VALUE.K_PAR PARAM_VALUE.K_PAR } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.K_PAR}] ${MODELPARAM_VALUE.K_PAR}
}

proc update_MODELPARAM_VALUE.A_WIDTH { MODELPARAM_VALUE.A_WIDTH PARAM_VALUE.A_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.A_WIDTH}] ${MODELPARAM_VALUE.A_WIDTH}
}

proc update_MODELPARAM_VALUE.TREE_V_MAX { MODELPARAM_VALUE.TREE_V_MAX PARAM_VALUE.TREE_V_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.TREE_V_MAX}] ${MODELPARAM_VALUE.TREE_V_MAX}
}

proc update_MODELPARAM_VALUE.BOOT_AXI_ADDR_W { MODELPARAM_VALUE.BOOT_AXI_ADDR_W PARAM_VALUE.BOOT_AXI_ADDR_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BOOT_AXI_ADDR_W}] ${MODELPARAM_VALUE.BOOT_AXI_ADDR_W}
}

proc update_MODELPARAM_VALUE.BOOT_AXI_DATA_W { MODELPARAM_VALUE.BOOT_AXI_DATA_W PARAM_VALUE.BOOT_AXI_DATA_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BOOT_AXI_DATA_W}] ${MODELPARAM_VALUE.BOOT_AXI_DATA_W}
}

proc update_MODELPARAM_VALUE.BOOT_AXI_ID_W { MODELPARAM_VALUE.BOOT_AXI_ID_W PARAM_VALUE.BOOT_AXI_ID_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BOOT_AXI_ID_W}] ${MODELPARAM_VALUE.BOOT_AXI_ID_W}
}

proc update_MODELPARAM_VALUE.BOOT_BURST_LEN { MODELPARAM_VALUE.BOOT_BURST_LEN PARAM_VALUE.BOOT_BURST_LEN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BOOT_BURST_LEN}] ${MODELPARAM_VALUE.BOOT_BURST_LEN}
}

proc update_MODELPARAM_VALUE.OUT_ABI_V24 { MODELPARAM_VALUE.OUT_ABI_V24 PARAM_VALUE.OUT_ABI_V24 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.OUT_ABI_V24}] ${MODELPARAM_VALUE.OUT_ABI_V24}
}

proc update_MODELPARAM_VALUE.OUT_TOP_K { MODELPARAM_VALUE.OUT_TOP_K PARAM_VALUE.OUT_TOP_K } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.OUT_TOP_K}] ${MODELPARAM_VALUE.OUT_TOP_K}
}

proc update_MODELPARAM_VALUE.OUT_PARENT_SENTINEL { MODELPARAM_VALUE.OUT_PARENT_SENTINEL PARAM_VALUE.OUT_PARENT_SENTINEL } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.OUT_PARENT_SENTINEL}] ${MODELPARAM_VALUE.OUT_PARENT_SENTINEL}
}

proc update_MODELPARAM_VALUE.CONNECT_SEAMS { MODELPARAM_VALUE.CONNECT_SEAMS PARAM_VALUE.CONNECT_SEAMS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONNECT_SEAMS}] ${MODELPARAM_VALUE.CONNECT_SEAMS}
}

proc update_MODELPARAM_VALUE.TAIL_FP16 { MODELPARAM_VALUE.TAIL_FP16 PARAM_VALUE.TAIL_FP16 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.TAIL_FP16}] ${MODELPARAM_VALUE.TAIL_FP16}
}

proc update_MODELPARAM_VALUE.TREE_P { MODELPARAM_VALUE.TREE_P PARAM_VALUE.TREE_P } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.TREE_P}] ${MODELPARAM_VALUE.TREE_P}
}

proc update_MODELPARAM_VALUE.TREE_PP { MODELPARAM_VALUE.TREE_PP PARAM_VALUE.TREE_PP } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.TREE_PP}] ${MODELPARAM_VALUE.TREE_PP}
}

proc update_MODELPARAM_VALUE.GEMV_KPAR64 { MODELPARAM_VALUE.GEMV_KPAR64 PARAM_VALUE.GEMV_KPAR64 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.GEMV_KPAR64}] ${MODELPARAM_VALUE.GEMV_KPAR64}
}

proc update_MODELPARAM_VALUE.GEMV_SCALED { MODELPARAM_VALUE.GEMV_SCALED PARAM_VALUE.GEMV_SCALED } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.GEMV_SCALED}] ${MODELPARAM_VALUE.GEMV_SCALED}
}

proc update_MODELPARAM_VALUE.GEMV_SCALE_FRAC { MODELPARAM_VALUE.GEMV_SCALE_FRAC PARAM_VALUE.GEMV_SCALE_FRAC } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.GEMV_SCALE_FRAC}] ${MODELPARAM_VALUE.GEMV_SCALE_FRAC}
}

proc update_MODELPARAM_VALUE.GPTQ_QZERO_RAW { MODELPARAM_VALUE.GPTQ_QZERO_RAW PARAM_VALUE.GPTQ_QZERO_RAW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.GPTQ_QZERO_RAW}] ${MODELPARAM_VALUE.GPTQ_QZERO_RAW}
}

proc update_MODELPARAM_VALUE.SEAM_R1_FULLW { MODELPARAM_VALUE.SEAM_R1_FULLW PARAM_VALUE.SEAM_R1_FULLW } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SEAM_R1_FULLW}] ${MODELPARAM_VALUE.SEAM_R1_FULLW}
}

proc update_MODELPARAM_VALUE.S7A_FRONT_SEQ { MODELPARAM_VALUE.S7A_FRONT_SEQ PARAM_VALUE.S7A_FRONT_SEQ } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.S7A_FRONT_SEQ}] ${MODELPARAM_VALUE.S7A_FRONT_SEQ}
}

proc update_MODELPARAM_VALUE.BACK_SEQ { MODELPARAM_VALUE.BACK_SEQ PARAM_VALUE.BACK_SEQ } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BACK_SEQ}] ${MODELPARAM_VALUE.BACK_SEQ}
}

proc update_MODELPARAM_VALUE.SLR_SAFE { MODELPARAM_VALUE.SLR_SAFE PARAM_VALUE.SLR_SAFE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SLR_SAFE}] ${MODELPARAM_VALUE.SLR_SAFE}
}

proc update_MODELPARAM_VALUE.ARR_GROUPS { MODELPARAM_VALUE.ARR_GROUPS PARAM_VALUE.ARR_GROUPS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ARR_GROUPS}] ${MODELPARAM_VALUE.ARR_GROUPS}
}

proc update_MODELPARAM_VALUE.ARR_PIPE_STAGES { MODELPARAM_VALUE.ARR_PIPE_STAGES PARAM_VALUE.ARR_PIPE_STAGES } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ARR_PIPE_STAGES}] ${MODELPARAM_VALUE.ARR_PIPE_STAGES}
}

proc update_MODELPARAM_VALUE.SEAM_FF { MODELPARAM_VALUE.SEAM_FF PARAM_VALUE.SEAM_FF } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SEAM_FF}] ${MODELPARAM_VALUE.SEAM_FF}
}

proc update_MODELPARAM_VALUE.FF_ACT_CACHE { MODELPARAM_VALUE.FF_ACT_CACHE PARAM_VALUE.FF_ACT_CACHE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FF_ACT_CACHE}] ${MODELPARAM_VALUE.FF_ACT_CACHE}
}

proc update_MODELPARAM_VALUE.DUAL_PORT { MODELPARAM_VALUE.DUAL_PORT PARAM_VALUE.DUAL_PORT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DUAL_PORT}] ${MODELPARAM_VALUE.DUAL_PORT}
}

proc update_MODELPARAM_VALUE.BODY_BATCH_B { MODELPARAM_VALUE.BODY_BATCH_B PARAM_VALUE.BODY_BATCH_B } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BODY_BATCH_B}] ${MODELPARAM_VALUE.BODY_BATCH_B}
}

proc update_MODELPARAM_VALUE.SEAM_FF_LAND_P { MODELPARAM_VALUE.SEAM_FF_LAND_P PARAM_VALUE.SEAM_FF_LAND_P } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SEAM_FF_LAND_P}] ${MODELPARAM_VALUE.SEAM_FF_LAND_P}
}

proc update_MODELPARAM_VALUE.TREE_SEAM_FF { MODELPARAM_VALUE.TREE_SEAM_FF PARAM_VALUE.TREE_SEAM_FF } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.TREE_SEAM_FF}] ${MODELPARAM_VALUE.TREE_SEAM_FF}
}

proc update_MODELPARAM_VALUE.LMDOWN_MEM_DIR_P { MODELPARAM_VALUE.LMDOWN_MEM_DIR_P PARAM_VALUE.LMDOWN_MEM_DIR_P } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.LMDOWN_MEM_DIR_P}] ${MODELPARAM_VALUE.LMDOWN_MEM_DIR_P}
}

proc update_MODELPARAM_VALUE.D1_LAT { MODELPARAM_VALUE.D1_LAT PARAM_VALUE.D1_LAT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.D1_LAT}] ${MODELPARAM_VALUE.D1_LAT}
}

proc update_MODELPARAM_VALUE.PREFETCH_COL_P { MODELPARAM_VALUE.PREFETCH_COL_P PARAM_VALUE.PREFETCH_COL_P } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PREFETCH_COL_P}] ${MODELPARAM_VALUE.PREFETCH_COL_P}
}

proc update_MODELPARAM_VALUE.D1_RDLAT { MODELPARAM_VALUE.D1_RDLAT PARAM_VALUE.D1_RDLAT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.D1_RDLAT}] ${MODELPARAM_VALUE.D1_RDLAT}
}

proc update_MODELPARAM_VALUE.D1_WLAT { MODELPARAM_VALUE.D1_WLAT PARAM_VALUE.D1_WLAT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.D1_WLAT}] ${MODELPARAM_VALUE.D1_WLAT}
}

proc update_MODELPARAM_VALUE.R4_BANKSPLIT { MODELPARAM_VALUE.R4_BANKSPLIT PARAM_VALUE.R4_BANKSPLIT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.R4_BANKSPLIT}] ${MODELPARAM_VALUE.R4_BANKSPLIT}
}

proc update_MODELPARAM_VALUE.CTX_WIN { MODELPARAM_VALUE.CTX_WIN PARAM_VALUE.CTX_WIN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CTX_WIN}] ${MODELPARAM_VALUE.CTX_WIN}
}

proc update_MODELPARAM_VALUE.DEPTH_N { MODELPARAM_VALUE.DEPTH_N PARAM_VALUE.DEPTH_N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DEPTH_N}] ${MODELPARAM_VALUE.DEPTH_N}
}

proc update_MODELPARAM_VALUE.EMBG_EN { MODELPARAM_VALUE.EMBG_EN PARAM_VALUE.EMBG_EN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.EMBG_EN}] ${MODELPARAM_VALUE.EMBG_EN}
}

proc update_MODELPARAM_VALUE.EMB_NCH { MODELPARAM_VALUE.EMB_NCH PARAM_VALUE.EMB_NCH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.EMB_NCH}] ${MODELPARAM_VALUE.EMB_NCH}
}

proc update_MODELPARAM_VALUE.FORREASON { MODELPARAM_VALUE.FORREASON PARAM_VALUE.FORREASON } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FORREASON}] ${MODELPARAM_VALUE.FORREASON}
}

proc update_MODELPARAM_VALUE.FR_KF_BANKS { MODELPARAM_VALUE.FR_KF_BANKS PARAM_VALUE.FR_KF_BANKS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FR_KF_BANKS}] ${MODELPARAM_VALUE.FR_KF_BANKS}
}

proc update_MODELPARAM_VALUE.FR_ABANK_BRAM { MODELPARAM_VALUE.FR_ABANK_BRAM PARAM_VALUE.FR_ABANK_BRAM } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FR_ABANK_BRAM}] ${MODELPARAM_VALUE.FR_ABANK_BRAM}
}

proc update_MODELPARAM_VALUE.HEAD_PAR { MODELPARAM_VALUE.HEAD_PAR PARAM_VALUE.HEAD_PAR } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.HEAD_PAR}] ${MODELPARAM_VALUE.HEAD_PAR}
}

proc update_MODELPARAM_VALUE.ACCFIX_SEAM_P { MODELPARAM_VALUE.ACCFIX_SEAM_P PARAM_VALUE.ACCFIX_SEAM_P } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ACCFIX_SEAM_P}] ${MODELPARAM_VALUE.ACCFIX_SEAM_P}
}

proc update_MODELPARAM_VALUE.ACCFIX_B2P_P { MODELPARAM_VALUE.ACCFIX_B2P_P PARAM_VALUE.ACCFIX_B2P_P } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ACCFIX_B2P_P}] ${MODELPARAM_VALUE.ACCFIX_B2P_P}
}

proc update_MODELPARAM_VALUE.ACCFIX_KC_P { MODELPARAM_VALUE.ACCFIX_KC_P PARAM_VALUE.ACCFIX_KC_P } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ACCFIX_KC_P}] ${MODELPARAM_VALUE.ACCFIX_KC_P}
}

proc update_MODELPARAM_VALUE.ACCFIX_KVONLY_P { MODELPARAM_VALUE.ACCFIX_KVONLY_P PARAM_VALUE.ACCFIX_KVONLY_P } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ACCFIX_KVONLY_P}] ${MODELPARAM_VALUE.ACCFIX_KVONLY_P}
}

proc update_MODELPARAM_VALUE.OBS_LVL_P { MODELPARAM_VALUE.OBS_LVL_P PARAM_VALUE.OBS_LVL_P } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.OBS_LVL_P}] ${MODELPARAM_VALUE.OBS_LVL_P}
}

proc update_MODELPARAM_VALUE.ENORM_D0 { MODELPARAM_VALUE.ENORM_D0 PARAM_VALUE.ENORM_D0 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ENORM_D0}] ${MODELPARAM_VALUE.ENORM_D0}
}

proc update_MODELPARAM_VALUE.PRECISION_SAFE { MODELPARAM_VALUE.PRECISION_SAFE PARAM_VALUE.PRECISION_SAFE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PRECISION_SAFE}] ${MODELPARAM_VALUE.PRECISION_SAFE}
}

proc update_MODELPARAM_VALUE.SILU_RANGE_FIX_P { MODELPARAM_VALUE.SILU_RANGE_FIX_P PARAM_VALUE.SILU_RANGE_FIX_P } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SILU_RANGE_FIX_P}] ${MODELPARAM_VALUE.SILU_RANGE_FIX_P}
}

proc update_MODELPARAM_VALUE.KV_V_RAM_STYLE { MODELPARAM_VALUE.KV_V_RAM_STYLE PARAM_VALUE.KV_V_RAM_STYLE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.KV_V_RAM_STYLE}] ${MODELPARAM_VALUE.KV_V_RAM_STYLE}
}

proc update_MODELPARAM_VALUE.KV_K_RAM_STYLE { MODELPARAM_VALUE.KV_K_RAM_STYLE PARAM_VALUE.KV_K_RAM_STYLE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.KV_K_RAM_STYLE}] ${MODELPARAM_VALUE.KV_K_RAM_STYLE}
}

proc update_MODELPARAM_VALUE.COMBINE_LOCAL { MODELPARAM_VALUE.COMBINE_LOCAL PARAM_VALUE.COMBINE_LOCAL } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.COMBINE_LOCAL}] ${MODELPARAM_VALUE.COMBINE_LOCAL}
}

proc update_MODELPARAM_VALUE.REQUANT_HEADMUX { MODELPARAM_VALUE.REQUANT_HEADMUX PARAM_VALUE.REQUANT_HEADMUX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.REQUANT_HEADMUX}] ${MODELPARAM_VALUE.REQUANT_HEADMUX}
}

proc update_MODELPARAM_VALUE.ROPE_CS_PIPE { MODELPARAM_VALUE.ROPE_CS_PIPE PARAM_VALUE.ROPE_CS_PIPE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ROPE_CS_PIPE}] ${MODELPARAM_VALUE.ROPE_CS_PIPE}
}

proc update_MODELPARAM_VALUE.ROPE_POS_REP { MODELPARAM_VALUE.ROPE_POS_REP PARAM_VALUE.ROPE_POS_REP } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ROPE_POS_REP}] ${MODELPARAM_VALUE.ROPE_POS_REP}
}

proc update_MODELPARAM_VALUE.KV_ADDR_REP { MODELPARAM_VALUE.KV_ADDR_REP PARAM_VALUE.KV_ADDR_REP } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.KV_ADDR_REP}] ${MODELPARAM_VALUE.KV_ADDR_REP}
}

proc update_MODELPARAM_VALUE.KV_RES_CLR_REP { MODELPARAM_VALUE.KV_RES_CLR_REP PARAM_VALUE.KV_RES_CLR_REP } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.KV_RES_CLR_REP}] ${MODELPARAM_VALUE.KV_RES_CLR_REP}
}

proc update_MODELPARAM_VALUE.G8_SPLICE { MODELPARAM_VALUE.G8_SPLICE PARAM_VALUE.G8_SPLICE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_SPLICE}] ${MODELPARAM_VALUE.G8_SPLICE}
}

proc update_MODELPARAM_VALUE.G8_K_C { MODELPARAM_VALUE.G8_K_C PARAM_VALUE.G8_K_C } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_K_C}] ${MODELPARAM_VALUE.G8_K_C}
}

proc update_MODELPARAM_VALUE.G8_MAX_BURST { MODELPARAM_VALUE.G8_MAX_BURST PARAM_VALUE.G8_MAX_BURST } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_MAX_BURST}] ${MODELPARAM_VALUE.G8_MAX_BURST}
}

proc update_MODELPARAM_VALUE.G8_PIPELINED { MODELPARAM_VALUE.G8_PIPELINED PARAM_VALUE.G8_PIPELINED } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_PIPELINED}] ${MODELPARAM_VALUE.G8_PIPELINED}
}

proc update_MODELPARAM_VALUE.G8_OUTSTANDING { MODELPARAM_VALUE.G8_OUTSTANDING PARAM_VALUE.G8_OUTSTANDING } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_OUTSTANDING}] ${MODELPARAM_VALUE.G8_OUTSTANDING}
}

proc update_MODELPARAM_VALUE.G8_ROWPP { MODELPARAM_VALUE.G8_ROWPP PARAM_VALUE.G8_ROWPP } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_ROWPP}] ${MODELPARAM_VALUE.G8_ROWPP}
}

proc update_MODELPARAM_VALUE.G8_PREFETCH { MODELPARAM_VALUE.G8_PREFETCH PARAM_VALUE.G8_PREFETCH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_PREFETCH}] ${MODELPARAM_VALUE.G8_PREFETCH}
}

proc update_MODELPARAM_VALUE.G8_DATA_W { MODELPARAM_VALUE.G8_DATA_W PARAM_VALUE.G8_DATA_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_DATA_W}] ${MODELPARAM_VALUE.G8_DATA_W}
}

proc update_MODELPARAM_VALUE.G8_OPT1C { MODELPARAM_VALUE.G8_OPT1C PARAM_VALUE.G8_OPT1C } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_OPT1C}] ${MODELPARAM_VALUE.G8_OPT1C}
}

proc update_MODELPARAM_VALUE.G8_OPT1D { MODELPARAM_VALUE.G8_OPT1D PARAM_VALUE.G8_OPT1D } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_OPT1D}] ${MODELPARAM_VALUE.G8_OPT1D}
}

proc update_MODELPARAM_VALUE.G8_MIN_NG { MODELPARAM_VALUE.G8_MIN_NG PARAM_VALUE.G8_MIN_NG } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_MIN_NG}] ${MODELPARAM_VALUE.G8_MIN_NG}
}

proc update_MODELPARAM_VALUE.G8_OPT1F { MODELPARAM_VALUE.G8_OPT1F PARAM_VALUE.G8_OPT1F } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_OPT1F}] ${MODELPARAM_VALUE.G8_OPT1F}
}

proc update_MODELPARAM_VALUE.G8_SELPIPE { MODELPARAM_VALUE.G8_SELPIPE PARAM_VALUE.G8_SELPIPE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_SELPIPE}] ${MODELPARAM_VALUE.G8_SELPIPE}
}

proc update_MODELPARAM_VALUE.G8_OPT1G { MODELPARAM_VALUE.G8_OPT1G PARAM_VALUE.G8_OPT1G } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_OPT1G}] ${MODELPARAM_VALUE.G8_OPT1G}
}

proc update_MODELPARAM_VALUE.G8_WIDE { MODELPARAM_VALUE.G8_WIDE PARAM_VALUE.G8_WIDE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_WIDE}] ${MODELPARAM_VALUE.G8_WIDE}
}

proc update_MODELPARAM_VALUE.G8_WIDE_P { MODELPARAM_VALUE.G8_WIDE_P PARAM_VALUE.G8_WIDE_P } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_WIDE_P}] ${MODELPARAM_VALUE.G8_WIDE_P}
}

proc update_MODELPARAM_VALUE.G8_MROUND { MODELPARAM_VALUE.G8_MROUND PARAM_VALUE.G8_MROUND } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_MROUND}] ${MODELPARAM_VALUE.G8_MROUND}
}

proc update_MODELPARAM_VALUE.G8_SCALAR_II1 { MODELPARAM_VALUE.G8_SCALAR_II1 PARAM_VALUE.G8_SCALAR_II1 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_SCALAR_II1}] ${MODELPARAM_VALUE.G8_SCALAR_II1}
}

proc update_MODELPARAM_VALUE.G8_CONSUME_HARDEN { MODELPARAM_VALUE.G8_CONSUME_HARDEN PARAM_VALUE.G8_CONSUME_HARDEN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_CONSUME_HARDEN}] ${MODELPARAM_VALUE.G8_CONSUME_HARDEN}
}

proc update_MODELPARAM_VALUE.G8_LANE_FIX { MODELPARAM_VALUE.G8_LANE_FIX PARAM_VALUE.G8_LANE_FIX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_LANE_FIX}] ${MODELPARAM_VALUE.G8_LANE_FIX}
}

proc update_MODELPARAM_VALUE.G8_WIDE_IL { MODELPARAM_VALUE.G8_WIDE_IL PARAM_VALUE.G8_WIDE_IL } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_WIDE_IL}] ${MODELPARAM_VALUE.G8_WIDE_IL}
}

proc update_MODELPARAM_VALUE.G8_WIDE_M { MODELPARAM_VALUE.G8_WIDE_M PARAM_VALUE.G8_WIDE_M } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_WIDE_M}] ${MODELPARAM_VALUE.G8_WIDE_M}
}

proc update_MODELPARAM_VALUE.G8_NCH { MODELPARAM_VALUE.G8_NCH PARAM_VALUE.G8_NCH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.G8_NCH}] ${MODELPARAM_VALUE.G8_NCH}
}

proc update_MODELPARAM_VALUE.BUFRDREG { MODELPARAM_VALUE.BUFRDREG PARAM_VALUE.BUFRDREG } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BUFRDREG}] ${MODELPARAM_VALUE.BUFRDREG}
}

proc update_MODELPARAM_VALUE.STAGE_INJECT { MODELPARAM_VALUE.STAGE_INJECT PARAM_VALUE.STAGE_INJECT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STAGE_INJECT}] ${MODELPARAM_VALUE.STAGE_INJECT}
}

