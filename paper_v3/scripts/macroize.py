#!/usr/bin/env python3
"""Replace literal result numbers in the sections by generated macros (macros_main, macros_x, macros_r1).
Each pair is (old, new, expected count); the script refuses to write if a count differs."""
from pathlib import Path
SEC = Path(__file__).resolve().parent.parent / 'sec'
R = {
'0_abstract.tex': [
 ('In about 187,000 requests to 12 VLMs, none of 40 predicted orderings', 'In about \\CscaleCalls{} requests to \\CscaleModels{} VLMs, none of \\XasN{} predicted orderings', 1),
 ('raised it by up to 0.28, from 0.62 to 0.90.', 'raised it by up to \\XcsMax, from \\CtLsGemThreeEightRecW{} to \\CtLsGemThreeEightRecR.', 1),
],
'1_intro.tex': [
 ('(a) An $11{,}236 \\times 7{,}698$ px sheet sent whole within 2{,}500 tokens of 32 px: one of its 23 targets spans $S = 0.40$ tokens',
  '(a) An $\\CfigW \\times \\CfigH$ px sheet sent whole within 2{,}500 tokens of 32 px: one of its \\CfigTargets{} targets spans $S = \\CfigSwhole$ tokens', 1),
 ('for GPT-5.6 recall falls from 0.77 to 0.47, although a packed call must report a median of only four targets.',
  'for GPT-5.6 recall falls from \\XmosRgptFiveSixkOne{} to \\XmosRgptFiveSixkEight, although a packed call must report a median of only \\CpackMedian{} targets.', 1),
 ("from $S = 2.86$ to 1.40; GPT-5.4's recall falls from 1.00 to 0.75.", "from $S = \\CsmallSw$ to \\CsmallSt; GPT-5.4's recall falls from \\CsmallRecWhole{} to \\CsmallRecTile.", 1),
 ('(d) The rule of \\cref{sec:safe}: 20 views at $S = 1.5$ that overlap by one target, 49{,}000 tokens in total, against a minimum of about 37{,}000',
  '(d) The rule of \\cref{sec:safe}: \\CfigViews{} views at $S = 1.5$ that overlap by one target, \\CfigTokens{} tokens in total, against a minimum of about \\CfigCmin', 1),
 ('An $11{,}236 \\times 7{,}698$ px drawing sent whole within 2{,}500 tokens of 32 px shrinks its 75 px symbols to about 13 px, 40\\% of a token ($S = 0.40$,',
  'An $\\CfigW \\times \\CfigH$ px drawing sent whole within 2{,}500 tokens of 32 px shrinks its \\CfigM{} px symbols to about \\CfigMsent{} px, \\CfigPctToken\\% of a token ($S = \\CfigSwhole$,', 1),
 ('the drawing above needs at least about 37{,}000 tokens at $S = 1.5$', 'the drawing above needs at least about \\CfigCmin{} tokens at $S = 1.5$', 1),
 ('yet F1 falls from 0.89 to 0.33;', 'yet F1 falls from \\XnFAOneRgptFiveSixSheets{} to \\XnFNATRgptFiveSixSheets;', 1),
 ('raises $S$ on the large drawing from 0.40 to about 1.4 but lowers it on a $2{,}489 \\times 1{,}301$ px drawing from 2.86 to 1.40, and on this sheet GPT-5.4\'s recall falls from 1.00 to 0.75',
  'raises $S$ on the large drawing from \\CfigSwhole{} to about 1.4 but lowers it on a $\\CsmallW \\times \\CsmallH$ px drawing from \\CsmallSw{} to \\CsmallSt, and on this sheet GPT-5.4\'s recall falls from \\CsmallRecWhole{} to \\CsmallRecTile', 1),
 ('it lowered recall in 19 of 21 model and data cells, significantly in 12.', 'it lowered recall in \\XtcOutNegPoint{} of \\XtcOutN{} model and data cells, significantly in \\XtcOutNeg.', 1),
 ('We test them in about 187{,}000 requests to 12 models,', 'We test them in about \\CscaleCalls{} requests to \\CscaleModels{} models,', 1),
 ('GPT-6.1 (GPT-5.6) & 8{,}192 & about 60 &', 'GPT-6.1 (GPT-5.6) & 8{,}192 & about \\CthumbSideGpt{} &', 1),
 ('Claude Opus 5.5 & 4{,}784 & about 46 &', 'Claude Opus 5.5 & 4{,}784 & about \\CthumbSideClaude{} &', 1),
 ('Gemini 4 (Gemini 3.8) & about 1{,}090 & about 22 &', 'Gemini 4 (Gemini 3.8) & about 1{,}090 & about \\CthumbSideGemini{} &', 1),
 ('Qwen3-VL & about 2{,}500 & about 33 &', 'Qwen3-VL & about 2{,}500 & about \\CthumbSideQwen{} &', 1),
 ('splits that never zoom out raised recall by 0.10 to 0.46 on average where $S < 1.5$, and splits that zoom out lowered it by 0.01 to 0.22',
  'splits that never zoom out raised recall by \\CmapBelowMin{} to \\CmapBelowMax{} on average where $S < 1.5$, and splits that zoom out lowered it by \\CmapZoomMin{} to \\CmapZoomMax', 1),
 ('In ten model and data settings', 'In \\CtCells{} model and data settings', 0),
 ('in ten model and data settings the rule', 'in \\CtCells{} model and data settings the rule', 1),
 ('(40 architectural drawings, 1{,}028 instances of legend symbols)', '(40 architectural drawings, 1{,}028 instances of legend symbols)', 1),
],
'3_protocol.tex': [
 ('We use 3{,}398 images of four kinds:', 'We use \\CscaleImages{} images of four kinds:', 1),
 ('and 2{,}769 controlled images,', 'and \\CscaleCtrl{} controlled images,', 1),
 ('We use 12 models from five families', 'We use \\CscaleModels{} models from five families', 1),
],
'4_why.tex': [
 ('(b) gives about 35{,}000 tokens at $S = 1.5$, and about 37{,}000 with the overlap margins', '(b) gives about \\CfigPropB{} tokens at $S = 1.5$, and about \\CfigCmin{} with the overlap margins', 1),
 ('recall of 8 px symbols at its three budgets is 0.44, 0.54 and 0.71;', 'recall of 8 px symbols at its three budgets is \\CbudEightLo, \\CbudEightDef{} and \\CbudEightUh;', 1),
 ('adds 0.17 F1 on items whose $S$ is below 1.5 and 0.05 on the others.', 'adds \\XfTOnesheetssmallDL{} F1 on items whose $S$ is below 1.5 and \\XfTOnesheetslargeDL{} on the others.', 1),
 ('Qwen3-VL recalls 0.22 of 96 targets', 'Qwen3-VL recalls \\XcapNlastqwenThreevl{} of 96 targets', 1),
 ('lowers recall from 0.77 to 0.47 (difference $-0.31$, interval $-0.44$ to $-0.16$), although a packed call must report a median of only four targets (mean 8.6, at most 35).',
  'lowers recall from \\XmosRgptFiveSixkOne{} to \\XmosRgptFiveSixkEight{} (difference \\XmosdRgptFiveSixkEight, interval \\XmosdRCIgptFiveSixkEight), although a packed call must report a median of only \\CpackMedian{} targets (mean \\CpackMean, at most \\CpackMax).', 1),
 ('in the controlled sweep 335 of 336 calls ended normally, and the longest answer used 3{,}428 tokens of the 8{,}192 allowed.',
  'in the controlled sweep \\CcapNormal{} of \\CcapCalls{} calls ended normally, and the longest answer used \\CcapMaxOut{} tokens of the 8{,}192 allowed.', 1),
 ('Gemini 3.8 recalls 0.58 more than Gemini 2.5;', 'Gemini 3.8 recalls \\XtOneHThree{} more than Gemini 2.5;', 1),
 ('the $S$ it needs for recall 0.9 falls from 0.76 to 0.29', 'the $S$ it needs for recall 0.9 falls from \\CgenSninetyGemThreeEight{} to \\CgenSninetyGemFour', 1),
 ('recall rises from 0.85 (GPT-5.4) to 0.99 (GPT-5.6) and from 0.57 (Sonnet 4.6) to 1.00 (Sonnet 5.5,',
  'recall rises from \\XcapNlastgptFiveFour{} (GPT-5.4) to \\XcapNlastgptFiveSix{} (GPT-5.6) and from \\XcapNlastsonnetFourSix{} (Sonnet 4.6) to \\XcapNlastsonnetFiveFive{} (Sonnet 5.5,', 1),
 ('GPT-6.1 holds beyond 300 targets', 'GPT-6.1 holds at \\CgenGptSixOneMax{} targets', 1),
 ('while enlarging gave GPT-5.6 0.11.', 'while enlarging gave GPT-5.6 \\XnHThreegptFiveSixOVZOVC.', 1),
],
'5_safe.tex': [
 ('no cell is significantly below zero, the worst is $-0.04$,', 'no cell is significantly below zero, the worst is \\CmapWorst,', 1),
 ('$+0.26$ to $+0.46$ for $S < 0.5$,', '\\CmapLowSMin{} to \\CmapLowSMax{} for $S < 0.5$,', 1),
 ('it rises from $+0.07$ to $+0.17$.', 'it rises from \\CmapMidSMin{} to \\CmapMidSMax.', 1),
 ('Gemini 3.8, Gemma, Qwen and Claude ($+0.08$ to $+0.14$);', 'Gemini 3.8, Gemma, Qwen and Claude (\\CmapBandMin{} to \\CmapBandMax);', 1),
 ('lose in every cell, from $-0.01$ to $-0.22$, significantly in four of six.', 'lose in every cell, by \\CmapZoomMin{} to \\CmapZoomMax, significantly in \\CmapZoomSig{} of \\CmapZoomCells.', 1),
 ('None of the 40 orderings is violated and 24 hold significantly; the smallest difference is $-0.03$. At fixed $S$, less content raises recall by $+0.06$ to $+0.37$, significantly for 4 of 8 models',
  'None of the \\XasN{} orderings is violated and \\XasSup{} hold significantly; the smallest difference is \\XasMin. At fixed $S$, less content raises recall by \\XasThmMin{} to \\XasThmMax, significantly for \\XasThmSup{} of \\XasModels{} models', 1),
 ('Of the 89 pairs of calls that the assumptions order in our other experiments, 60 hold significantly and 4 are violated',
  'Of the \\XtcN{} pairs of calls that the assumptions order in our other experiments, \\XtcSup{} hold significantly and \\XtcViol{} are violated', 1),
 ('All four violations we found', 'All \\XtcViol{} violations we found', 1),
],
'6_practice.tex': [
 ('The smallest change is $-0.02$, and recall rose significantly in five settings: Gemini 3.8 on LegendSpot from 0.62 to 0.90 (45\\% relative), Gemini 4 on LegendSpot from 0.87 to 0.98 (12\\%), Qwen3-VL on LegendSpot from 0.37 to 0.56 (51\\%), Gemini 3.8 on floor plans from 0.25 to 0.43 (73\\%), and Qwen3-VL on floor plans from 0.05 to 0.12 (127\\%),',
  'The smallest change is \\CtMinChange, and recall rose significantly in \\CtSigRec{} settings: Gemini 3.8 on LegendSpot from \\CtLsGemThreeEightRecW{} to \\CtLsGemThreeEightRecR{} (\\CtLsGemThreeEightRel\\% relative), Gemini 4 on LegendSpot from \\CtLsGemFourRecW{} to \\CtLsGemFourRecR{} (\\CtLsGemFourRel\\%), Qwen3-VL on LegendSpot from \\CtLsQwenThreevlRecW{} to \\CtLsQwenThreevlRecR{} (\\CtLsQwenThreevlRel\\%), Gemini 3.8 on floor plans from \\CtFpGemThreeEightRecW{} to \\CtFpGemThreeEightRecR{} (\\CtFpGemThreeEightRel\\%), and Qwen3-VL on floor plans from \\CtFpQwenThreevlRecW{} to \\CtFpQwenThreevlRecR{} (\\CtFpQwenThreevlRel\\%),', 1),
 ('Precision rose significantly in six of ten settings', 'Precision rose significantly in \\CtSigPrec{} of \\CtCells{} settings', 1),
 ("The price is 3.1 to 8.6 times the whole call's tokens, 0.11 to 1.06 times the 44 px recipe's, and 1.6 to 3.5 times $\\Cmin$.",
  "The price is \\CtTokMin{} to \\CtTokMax{} times the whole call's tokens, \\CtTokDMin{} to \\CtTokDMax{} times the 44 px recipe's, and \\CtCminMin{} to \\CtCminMax{} times $\\Cmin$.", 1),
 ("49 of GPT-6.1's 60 floor plans need a single view.", "\\CtOneViewGpt{} of GPT-6.1's 60 floor plans need a single view.", 1),
 ('for Gemini 3.8 and Qwen3-VL ($-0.13$ and $-0.11$).', 'for Gemini 3.8 and Qwen3-VL (\\CtFpGemThreeEightdRecD{} and \\CtFpQwenThreevldRecD).', 1),
 ('shows only $-0.02$, not significant.', 'shows only \\CtFpGemFourdRecD, not significant.', 1),
 ('Template matching reaches 0.87 F1 on the 34 drawings, and the best model-configured VLM tiles reach that level (Claude Sonnet 5.5 0.92, Gemini 3.8 0.94).',
  'Template matching reaches \\FTMdevAll{} F1 on the 34 drawings, and the best model-configured VLM tiles reach that level (Claude Sonnet 5.5 \\FDsonnetFiveFiveAll, Gemini 3.8 \\FDgemThreeEightAll).', 1),
 ('template matching falls to 0.38 and the VLMs are at least 0.30 higher.', 'template matching falls to \\XrdFTM{} and the VLMs are at least 0.30 higher.', 1),
 ('raise small-object F1 for GPT-5.6 by 0.14 and for Claude Sonnet 5.5 by 0.20 (0.12 and 0.17 without truncated answers):',
  'raise small-object F1 for GPT-5.6 by \\XfscODgptFiveSixsmallTU{} and for Claude Sonnet 5.5 by \\XfscODsonnetFiveFivesmallTU{} (\\CfscNoTruncGptFiveSix{} and \\CfscNoTruncSonnetFiveFive{} without truncated answers):', 1),
 ("GPT-5.6's zoom agent reaches 0.97 F1 on full sheets against 0.91 for its tiles ($+0.06$, Holm $p = 0.013$); Claude Sonnet 5.5's agent reaches 0.95 against 0.90, not significant.",
  "GPT-5.6's zoom agent reaches \\XnFAGgptFiveSixSheets{} F1 on full sheets against \\XnFDRgptFiveSixSheets{} for its tiles (\\XnHOnegptFiveSix, Holm $p = \\XnXNineHOnegptFiveSixHolm$); Claude Sonnet 5.5's agent reaches \\XnFAGsonnetFiveFiveSheets{} against \\XnFDRsonnetFiveFiveSheets, not significant.", 1),
 ('enlarging the crops (raising $S$) gives GPT-5.6 $+0.11$, and native pixels add $+0.01$ for GPT-5.6 and $+0.09$ for Claude:',
  'enlarging the crops (raising $S$) gives GPT-5.6 \\XnHThreegptFiveSixOVZOVC, and native pixels add \\XnHThreegptFiveSixAGOVZ{} for GPT-5.6 and \\XnHThreesonnetFiveFiveAGOVZ{} for Claude:', 1),
 ('GPT-5.6 used 11.4 zooms per drawing', 'GPT-5.6 used \\XagZoomsgptFiveSix{} zooms per drawing', 1),
 ('the Gemini 3.8 and Qwen3-VL agents fail (0.64 and 0.21)', 'the Gemini 3.8 and Qwen3-VL agents fail (\\XagFgemThreeEightAll{} and \\XqAgF)', 1),
 ('ten model and data settings in all', '\\CtCells{} model and data settings in all', 1),
],
}
for fn, pairs in R.items():
    p = SEC / fn; t = p.read_text()
    for old, new, cnt in pairs:
        c = t.count(old)
        assert c == cnt, (fn, c, cnt, old[:70])
        if cnt: t = t.replace(old, new)
    p.write_text(t)
print('ok')
