
  

# ICTA: Hitri začetki z umetno inteligenco - praktični del

## Praktični detajli

### Programska oprema

Za delo na oddaljenem računalniku bomo uporabili dva programa: prvi je x2go, ki nam bo omogočil dostop do ukazne vrstice na strežniku, še pomembneje pa, da bo omogočil vizualizacijo na lokalnem računalniku.
Drugi program, WinSCP, bomo uporabili za prenos datotek z in na strežnik. Omogočil nam bo lokalno urejanje skript, ki jih bomo nato pognali v ukazni vrstici. Ker želimo narediti celo stvar precej bolj prenosljivo in enostavno, se kompleksnejših IDEjev ne bomo posluževali.

Večina višjenivojskega razvoja strojnega učenja se izvaja v programskem jeziku python. Python je zaradi fleksibilnosti primeren za hiter razvoj in testiranje idej, a je hkrati precej počasen. Veliko operacij in knjižnic za strojno učenje je zato napisanih v C/C++, njihove funkcije pa se nato kličejo preko pythona.

### CUDA

CUDA (Compute Unified Device Architecture) je vmesnik (API), ki omogoča uporabo grafičnih procesnih enot (GPU) za pospeševanje računanja. Je skoraj nepogrešljiva za stronjo učenje, saj lahko proces pospeši za 5-10x.

### Conda

Conda je package manager, ki ga uporabljamo za ločevanje okolij, v katerih izvajamo algoritme. Ker imajo različna ogrodja svoje zahteve za verzije knjižnic, lahko s condo ustvarjamo virtualna okolja, ki vsebujejo ustrezne verzije za posamezen projek.

### Strojna oprema

Za to delavnico boste uporabili računalnik iz laboratorija LSI, imenovan Neumann. Ima 128 CPU jeder, 1 TB RAMa in 4 Nvidia A100 GPUje (vsak ima po 80GB VRAMa). Vaš začasen dostop bo preko uporabniškega imena `user<zaporedna številka>` in ustreznega gesla. Neumann je dostopen na IP naslovu `192.168.91.195` preko ssh protokola.

## Veliki jezikovni modeli (angl. Large Language Models - LLM)

Te modele bomo poganjali v ogrodju ollama, ki omogoča enostavno uporabo različnih jezikovnih modelov.

https://github.com/ollama/ollama

Glavna prednost poganjanja jezikovnih modelov lokalno je, da potencialno zaupnih podatkov ne pošiljamo na oddaljene strežnike. Glavna slabost pa, da potrebujemo dovolj zmogljivo strojno opremo.

Modeli globokega učenja imajo svoje znanje shranjeno v utežeh (angl. weights), ki predstavljajo vrednosti parametrov. Ker je lahko (odvisno od kompleksnosti modela) število parametrov v milijardah, potrebujemo za uporabo takšnih modelov dovolj prostora na disku, še bolj pomembna pa je dovoljšna količina delovnega spomina na grafičnih karticah (angl. VRAM).

Seznam modelov, ki so na voljo v ogrodju ollama se nahaja tukaj: https://ollama.com/library
Modeli se med sabo razlikujejo po arhitekturi, viru podatkov in kompleksnosti. V splošnem so bolj kompleksni modeli bolj kvalitetni, a zaradi tega tudi večji in počasnejši.

Ogrodje ollama poženemo z ukazom `./start_ollama.sh <št. GPUja> <up. ime>`, nato pa `docker exec -it ollama-<up. ime> bash`. Odpre se nam nova ukazna vrstica, kjer lahko z ukazom `ollama run <ime modela>` model tudi poženemo. Najprej bomo preizkusili model [gemma3](https://ollama.com/library/gemma3), torej `ollama run gemma3:12b`.

Nekateri jezikovni modeli so multimodalni, torej vsebujejo  komponente, ki znajo delati z dodatnimi modalitetami, kot so na primer slike. V ogrodju ollama lahko v vprašanju podamo pot do slike, ki jo bo multimodalni model nato prebral in interpretiral. 

Nekaj primerov promptov za testiranje jezikovnih modelov:
- `Prosim, opiši mi, kaj se nahaja na sliki /home/data/cat2.jpg`
- `Please write down the main ideas of machine learning, specifically LLMs.`


## YOLO

Model YOLO (You Only Look Once) je eden bolj razširjenih detektorjev objektov. Ti modeli v osnovi pregledajo sliko in vrnejo seznam očrtanih pravokotnikov (angl. bounding box), ki predstavljajo objekte v sliki, ter seznam njihovih semantičnih razredov. Ti razredi so navadno vnaprej določeni in skladni z učnim procesom modela.

V zadnjih letih je organizacija Ultralytics poskrbela, da so modeli za detekcijo, kot je YOLO zelo enostavno dostopni in zelo pogosto delujejo v realnem času. V tem sklopu bomo pogledali, kateri modeli so na voljo, kako jih poženemo in kakšne rezultate vračajo.

V našem primeru bomo model YOLO in njegove različice poganjali s skripto `ultralytics_demos.py`. Skripto poženemo z ukazom `python ultralytics_demos.py <ime_različice>`, kjer je ime različice eden od nizov "detect", "segment", "pose", "world" in "SAM".

### YOLO

Osnovna različica modela YOLO izvaja detekcijo objekov 80 različnih razredov (ki so uporabljeni v podatkovni zbirki [COCO](https://cocodataset.org/#home). Model je zasnovan z enim samim prehodom slike in cilja na delovanje v realnem času. Razvitih je bilo več izboljšanih verzij modela s strani različnih avtorjev, najnovejši model, ki ga trenutno ponuja Ultralytics je YOLOv12.

### YOLO-seg

Segmentacijska različica modela YOLO poleg bounding boxa vrne tudi masko, ki zajema piksle zaznanega objekta. Ker se predikcije izvajajo na manjši resoluciji, je treba maske pred prikazom povečati na originalno resolucijo, zato včasih niso zelo natančne. To lahko prilagodimo z uporabo parametra imgsz v fukciji model.predict()

### YOLO-world

YOLO-world je t.i. open-vocabulary model, ki detekcij ne proizvaja na podlagi vnaprej določenih razredov, ampak poleg slike kot vhod sprejme tudi seznam tekstovnih opisov, ki predstavljajo objekte, ki jih želimo detektirati.

### YOLO-pose

https://docs.ultralytics.com/tasks/pose/

Modeli YOLO-pose so osredotočeni na detekcijo pozicije oseb v sliki. Kot svoj izhod poleg pozicije oseb vračajo tudi napovedi pozicij ključnih točk, kot so sklepi, okončine in oči. Rezultate takih modelov lahko uporabimo za detekcijo ali identifikacijo dejavnosti, 3d pozicije oseb ali avtomatsko ocenjevanje npr. športnih aktivnosti.

### Sledenje (angl. tracking)

Sledenje je niša računalniškega vida, ki skuša izbranemu objektu slediti znotraj sekvence zaporednih slik. To je lahko enostavnejši problem kot detekcija, a lahko zaradi premikanja objektov in kamere ter zakrivanja postane precej težaven. Taki algoritmi od uporabnika navadno zahtevajo očrtan pravokotnik, ki označuje objekt, ki mu želimo slediti. Algoritem nato za vsako od slik v sekvenci najde isti objekt in vrne njegovo pozicijo. Sledilni algoritmi se pogosto uporabljajo v kombinaciji z detektorji.

Preizkusili bomo algoritem TrackerVit, ki je implementiran v knjižnici OpenCV. Skripta za testiranje je `cv2_tracking_demo.py`.

## SAM

SAM (oz. SAM2) je temeljni model, ki so ga razvili pri FAIR (Facebook AI Research). Njegova kratica pomeni Segment Anything Model in temelji na segmentaciji preko ?pozivov? (angl. prompt). Model vhodno sliko najprej zakodira preko enkoderja, nato pa za predikcijo objektov na sliki uporabi uporabnikov poziv, ki je lahko v obliki bounding boxa, točke, ali teksta. Prompt nato omeji fokus modela in vrne rezultate. Model lahko deluje tudi brez promptov, v tem primeru vrne vse regije, ki bi lahko bile objekti.

## CLIP/BLIP

Modela CLIP (Contrastive Language–Image Pre-training) in BLIP (Bootstrapping Language-Image Pre-training) sta predstavnik a modelov, ki povezujejo jezikovne pristope s slikovnimi modeli. Naučena sta bila na parih slika+opis, njun cilj pa je bil proizvesti značilke, ki so si v latentnem prostoru podobne, ne glede na modaliteto. V osnovni obliki lahko take modele uporabljamo za klasifikacijo slik, pogosto pa se uporabljajo tudi v povezavi z drugimi modeli računalniškega vida. Odvisno od smeri se naloga imenuje *captioning* (predikcija opisa iz slike) oz. *grounding* (lokalizacija želenega objekta na podlagi slike). V uporabi skupaj z generativnimi modeli pa lahko CLIP/BLIP uporabimo tudi za generiranje slik z zeljeno vsebino.

## Temeljni modeli (angl. foundation models)

Ideja temeljnih modelov je, da postanejo z dovoljšnjo količino podatkov dovolj splošni, da lahko proizvedejo dobre rezultate tudi na domenah in v situacijah, ki jih med učnim procesom niso obravnavali. Temu procesu rečemo *zero-shot performance.* Primeri takih modelov so npr. SAM, GPT, CLIP in DINO.

Implementacija temeljnih modelov, ki jih bomo uporabili se nahaja v direktoriju `zap-it`. To ogrodje je strukturirano za avtomatsko segmentacijo poljubnih objektov. Za enkodiranje slik se uporabi model SAM, ki vrne segmentacijske maske za vse regije, ki jih model prepozna kot potencialne objekte. Nato se izračunajo značilke posamezne maske z vizualno-jezikovnim modelom CLIP, hkrati pa se podobne značilke izračunajo tudi iz tekstovnega opisa objektov. Za posamezno regijo se nato vektor značilk primerja z vektorjem opisa. Če je kosinusna razdalja med vektorjema dovolj majhna, lahko posamezno regijo z veliko verjetnostjo klasificiramo kot željeni objekt.

Za uporabo orodja moramo najprej aktivirati conda okolje z ukazom `conda activate zap-it`, nato pa metodo poženemo z ukazom `python zap-it-batch.py --config configs/example.yaml --dir path/to/images --verbose full` .

## HPC

### Ukazi
- salloc
	- izvede zahtevo za vire (CPU jedra, RAM, GPU) za določen čas
- srun
	- požene program na zahtevanem vozlišču
- scancel
	- prekliče alokacijo virov