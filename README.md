# Guida alla lettura del codice

Nella cartella __lib__ sono presenti i diversi moduli che contengono il codice per l'elaborazione:

- __pipeline__: qui si trova il codice che mette insieme le varie fasi dell'elaborazione
per produrre l'output desiderato, ovvero una sorta di scannerizzazione del documento fotografato.
- __utility__: qui si trovano diverse funzioni di utility che ho scritto nell'arco dello
sviluppo del progetto. La maggior parte servono ad estrarre statistiche da matrici.
Solo una minima parte di queste sono effettivamente utilizzate,
quindi non direi che qui si trovi molto codice interessante.
- __binarization__: qui si trova il codice per trasformare l'immagine di input in un'immagine binaria
in cui sono poste in risalto le regioni contenenti testo scritto.
- __image_statistics__: qui si trovano gli algoritmi utilizzati per calcolare
le statistiche locali dell'immagine in modo computazionalmente efficiente.
- __page_frame__: qui si trova il codice per estrarre dall'immagine principale il
rettangolo minimo che contiene il foglio fotografato. Questa sezione è quella di
più difficile lettura: durante l'esposizione pensavo di mostrare alcuni esempi
per renderne più chiaro il funzionamento.
- __corners__: questo modulo contiene del codice che è utilizzato all'interno
di __page_frame__ per scegliere i pixel che corrispondono agli angoli della
cornice contenente l'immagine. Anche questa parte non è di facilissima lettura.
- __pre_processing__: questo modulo è responsabile della fase di pre-processing che deve
predisporre l'immagine alle fasi successive dell'elaborazione.
