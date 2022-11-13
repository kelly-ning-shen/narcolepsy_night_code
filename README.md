# narcolepsy_night_code
Ning Shen from Fudan University, China

## Objective & Motivation
Using full-night PSG recordings to automatically detect patients with narcolepsy.
- 针对夜晚睡眠中非事件性的睡眠障碍 → 缺少事件标注 → sparse supervision
- 想要结合睡眠结构来诊断
- 挖掘睡眠的内在信息

## The Key method
1. proposed: multitask
  - narcolepsy detection (patients with narcolepsy, non-narcolepsy controls)
  - sleep staging (to address the sparse supervision)
2. baseline 1: one-phase (without sleep staging)
  - narcolepsy detection (patients with narcolepsy, non-narcolepsy controls)
3. baseline 2: two-phase (problem: error propagation) (I adopted it in my previous paper)
> [1]	N. Shen et al., "Towards an automatic narcolepsy detection on ambiguous sleep staging and sleep transition dynamics joint model," J. Neural Eng., vol. 19, no. 5, Oct 1 2022, Art no. 056009, doi: 10.1088/1741-2552/ac8c6b.
  - step 1: sleep staging
  - step 2: narcolepsy detection

### Dataset
CNC (78 recordings)
> Used for narcolepsy testing sample 2 in: <br>
> [2]	J. B. Stephansen et al., "Neural network analysis of sleep stages enables efficient diagnosis of narcolepsy," Nat. Commun., Article vol. 9, p. 15, Dec 2018, Art no. 5229, doi: 10.1038/s41467-018-07229-3.
- T1 NARCOLEPSY (55)
- NON-NARCOLEPSY CONTROL (23)
	
## The major findings
My assumption: multitask performs better (?)

## Contributions & Limitaions
### Contributions
Avoid error propagation existed in two-phase models.

### Limitations
Only cover one dataset.

## Potential research
Apply on more datasets from more places.
