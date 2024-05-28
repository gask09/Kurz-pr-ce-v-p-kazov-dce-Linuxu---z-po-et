# Repo pro uchování souborů na zápočet 

Repo obsahuje čtyři soubory, `metacentrum.sh`, spustitelný skript pro metacentrum, `morpho_dataset.py` a `morpho_analyzer.py`, soubory primárně určené na stažení dat pro `tagger_competition.py`, který obsahuje výpočetně náročnou neuronovou síť. Poslední tři soubory mají svou vlastní složku *tagger_competition*. 

Jádro celého modelu (neuronové sítě) jsou dvě vrstvy obousměrných LSTM (tj. do části první vrstvy LSTM vstupují trénovací data od prvního slova do konce, do zbylé části LSTM od posledního slova po první), následované dvěma mezi sebou nepropojenými konvolučními vrstvami, jedna pro předný výstup z LSTM, druhá pro zpětný. Výstup obou konvolučních sítí je sečten a poté poslán na finální FC vrstvu. 

Výstupem jsou váhy modelu (.pt soubor) a .txt soubor, který obsahuje pro každé slovo z testovacího datasetu mnoho morfologických vlastností slov (pád, rod aj.).

Na testovacím datasetu bylo dosaženo přesnosti 97,24 %, bez použití konvolučních vrstev 97,10 %.

## Jak donutit program fungovat

Po přenesení dat přes `scp` na frontend metacentra (spustitelný .sh skript pracuje s frontendem skirit.ics.muni.cz) je potřeba zarezervovat si výpočetní kapacitu. Pro trénování modelu je nejlepší použít GPU (trénování trvá zhruba 4 hodiny @ 1x nVidia Tesla T4). Výpočetní kapacitu si lze zamluvit pomocí příkazu: 
```
qsub -q GPU -l select=1:ncpus=1:ngpus=1:mem=32gb:scratch_local=16gb -N tagger metacentrum.sh 
```
Po spuštění úlohy bude poměrně dlouho trvat instalace balíčků pro virtuální prostředí pythonu (proto je také mimojiné třeba vyžádat si 32 GB operační paměti). Běh a stav úlohy lze sledovat zalogováním na přiřazený výpočetní klastr skrze `ssh`. STDOUT a STDERR se nacházejí ve složce `/var/spool/pbs/spool`. Zde je třeba otevřít pomocí `tail -f` `JOBID.OU` pro zobrazení průběhu instalace balíčků a `JOBID.ER` pro sledování trénování modelu.  
