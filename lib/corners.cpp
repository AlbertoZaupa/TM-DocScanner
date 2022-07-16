#include "corners.h"
#include <cstdlib>
#include <iostream>

/*
Quando durante l'inseguimento di contorni viene trovato un pixel che può corrispondere ad un angolo
del foglio, si registra anche quanto peso dare a tale pixel nel determinare ciascuna delle due coordinate
dell'angolo ricercato.
Un candidato ottenuto scorrendo l'immagine dall'alto verso il basso, ricercando rette orizzontali,
sarà molto influente nella scelta della coordinata y e non molto nella la scelta della coordinata x. L'esatto
opposto vale per un candidato ottenuto scorrendo l'immagine da sinistra a destra.
E' possibile che durante la ricerca di un angolo, la ricerca di contorni non dia nessun candidato. In questo
caso se non sono stati trovati candidati nè lungo l'asse x nè lungo l'asse y, viene scelto come candidato il
rispettivo estremo dell'immagine (ad esempio durante la ricerca dell'angolo in alto a sinistra verrebbe scelto
il pixel (0, 0) ).

Il codice per scegliere il miglior candidato tra quelli trovati durante l'estrazione dei contorni è utilizzato anche
una volta che i 4 angoli del foglio sono stati determinati, per determinare il rettangolo che contiene tutti gli angoli.
In questa scelta viene data massima influenza agli angoli ottenuti confrontando due candidati, poi vengono
considerati gli angoli ottenuti da un solo candidato ed infine gli angoli corrispondenti agli estremi della pagina. 
Per scegliere tra candidati con lo stesso livello di influenza, se ne prende o il massimo o il minimo, in base a quale
angolo del rettangolo si sta cercando.

Le funzioni pick_col e pick_row sono implementate come una tabella di verità costruita dalle variabili booleane che
rappresentano l'influenza da attribuire ai candidati nella scelta, rispettivamente, di coordinata y ed x dell'angolo.
Dunque la tabella di verità è a 4 variabili. Per ciascuna combinazione di valori viene descritto l'output desiderato.
*/

void CornerCandidate::pick_col(CornerCandidate c1, CornerCandidate c2, int (*col_discriminating_func) (int, int)) {
    if (!c1.col_confidence && !c2.col_confidence && !c1.row_confidence && !c2.row_confidence) return;
    else if (c1.col_confidence && c2.col_confidence) {
        col_confidence = true;
        col = col_discriminating_func(c1.col, c2.col);
    }
    else if (c1.col_confidence && !c2.col_confidence) {
        col_confidence = true;
        col = c1.col;
    }
    else if (!c1.col_confidence && c2.col_confidence) {
        col_confidence = true;
        col = c2.col;
    }
    else if (c2.row_confidence && c1.row_confidence) {
        col_confidence = false;
        col = col_discriminating_func(c1.col, c2.col);
    }
    else if (!c1.row_confidence && c2.row_confidence) {
        col_confidence = false;
        col = c2.col;
    }
    else if (c1.row_confidence && !c2.row_confidence) {
        col_confidence = false;
        col = c1.col;
    }
    else {
        std::cerr<<"page_frame.pick_col(): unexpected input combination\n";
        exit(1);
    }
}

/*
La funzione è del tutto analoga a pick_col.
*/

void CornerCandidate::pick_row(CornerCandidate c1, CornerCandidate c2, int (*row_discriminating_func) (int, int)) {
    if (!c1.col_confidence && !c2.col_confidence && !c1.row_confidence && !c2.row_confidence) return;
    else if (c1.row_confidence && c2.row_confidence) {
        row_confidence = true;
        row = row_discriminating_func(c1.row, c2.row);
    }
    else if (c1.row_confidence && !c2.row_confidence) {
        row_confidence = true;
        row = c1.row;
    }
    else if (!c1.row_confidence && c2.row_confidence) {
        row_confidence = true;
        row = c2.row;
    }
    else if (c2.col_confidence && c1.col_confidence) {
        row_confidence = false;
        row = row_discriminating_func(c1.row, c2.row);
    }
    else if (!c1.col_confidence && c2.col_confidence) {
        row_confidence = false;
        row = c2.row;
    }
    else if (c1.col_confidence && !c2.col_confidence) {
        row_confidence = false;
        row = c1.row;
    }
    else {
        std::cerr<<"page_frame.pick_col(): unexpected input combination\n";
        exit(1);
    }
}

void CornerCandidate::init(int col, int row, bool col_confidence, bool row_confidence) {
    this->col = col;
    this->row = row;
    this->col_confidence = col_confidence;
    this->row_confidence = row_confidence;
}

CornerCandidate::CornerCandidate(int col, int row, bool col_confidence, bool row_confidence) {
    this->col = col;
    this->row = row;
    this->col_confidence = col_confidence;
    this->row_confidence = row_confidence;
}
