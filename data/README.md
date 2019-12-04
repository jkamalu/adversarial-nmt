## Shuffling procedure
shuf --random-source=europarl-v7.fr-en.en europarl-v7.fr-en.en > europarl-v7.fr-en.en.shuf
shuf --random-source=europarl-v7.fr-en.en europarl-v7.fr-en.fr > europarl-v7.fr-en.fr.shuf

## Splitting procedure
EUROPARL_LINES=$(cat europarl-v7.fr-en.en | wc -l)
head -$((EUROPARL_LINES-100000)) europarl-v7.fr-en.en.shuf > europarl-v7.fr-en.en.train
head -$((EUROPARL_LINES-100000)) europarl-v7.fr-en.fr.shuf > europarl-v7.fr-en.fr.train
tail -100000 europarl-v7.fr-en.en.shuf > europarl-v7.fr-en.en.val
tail -100000 europarl-v7.fr-en.fr.shuf > europarl-v7.fr-en.fr.val
