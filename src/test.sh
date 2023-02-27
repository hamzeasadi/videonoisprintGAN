#!/bin/bash

scp -P 40022 -i ~/.ssh/id_ed25519_confirm hasadi@143.239.81.1:~/$1/$2.pt ~/python/videonoisprintGAN/data/model
`echo "======================== model ${2} ==========================" >> log.txt`
`python testing.py --mn ${2} >> log.txt`
`echo "==============================================================" >> log.txt`



