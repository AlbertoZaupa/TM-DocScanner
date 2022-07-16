#include "page_frame.h"
#include "opencv2/opencv.hpp"
#include "utility.h"
#include "corners.h"

using namespace cv;

int PageFrame::CHASE_DEPTH = 400;
int PageFrame::TOLERANCE_FACTOR = 20;
int PageFrame::ALLOW_SHIFT = 4;
int PageFrame::RUDIMENTARY_DEPTH = 200;

/*
 Questa funzione estrae il rettangolo contenente il foglio da scannerizzare ricercandone i 4 angoli.
 Per esempio, per ricercare l'angolo in alto a sinista, l'immagine viene attraversata partendo dal pixel nella
 posizione (0, 0), scorrendone le colonne di ciascuna riga alla ricerca di un pixel bianco (che corrisponde ad
 un punto posto in risalto dall'operazione precedente). Quando viene trovato un pixel bianco, la funzione
 "edge_chase" determina se il pixel appartiene ad un contorno del foglio cercando di percorrrere una linea
 di pixel bianchi diretta verso il basso, che parta da quel pixel.
 Se una linea di questo tipo viene trovata, il pixel diventa un candidato per l'angolo ricercato.
 Successivamente l'immagine viene nuovamente attraversata partendo dal pixel in (0, 0), questa volta però
 scorredone le righe, alla ricerca di un pixel bianco da cui abbia inizio un linea bianca diretta da destra a sinista.
 I candidati ottenuti scorrendo le righe e le colonne vengono confrontati tramite delle funzioni apposite del
 modulo corners.h, per determinare la posizione dell'angolo in alto a sinistra.
*/

Rect get_page_frame(const Mat &filtered_image) {
    // La ricerca degli angoli si arresta a metà dell'immagine, sotto l'ipotesi che il foglio da scannerizare si trovi
    // a cavallo, almeno in parte, dei quattro quadranti dell'immagine.
    int margin_search_x_bound = filtered_image.size[1] / 2;
    int margin_search_y_bound = filtered_image.size[0] / 2;

    // Vegono inizializzati i candidati per i 4 angoli dell'immagine
    CornerCandidate TL_corner(0, 0, false, false);
    CornerCandidate TR_corner(filtered_image.size[1] - 1, 0, false, false);
    CornerCandidate BL_corner(0, filtered_image.size[0] - 1, false, false);
    CornerCandidate BR_corner(filtered_image.size[1] - 1, filtered_image.size[0] - 1, false, false);

    // Vengono inizializzati i candidati per gli angoli trovati attraversando l'immagine in diretzione
    // Nord -> Sud e Ovest -> Est
    CornerCandidate X_corner(0, 0, false, false);
    CornerCandidate Y_corner(0, 0, false, false);
    bool found_margin = false;

    // Ricerca dell'angolo in alto a sinistra. Prima l'immagine viene attraversata in direzione Ovest -> Est, poi in
    // direzione Nord -> Sud. Quando l'immagine è attraversata da Ovest ad Est, la linea di pixel bianchi ricercata
    // va dall'alto verso il basso, mentre quando l'immagine è attraversata da Nord a Sud va da sinistra a destra.
    for (int row=0; row<margin_search_y_bound && !found_margin; ++row) {
        for (int col=0; col<margin_search_x_bound && !found_margin; ++col) {
            if (edge_chase(filtered_image, row, col, N_S)) {
                X_corner.row = row;
                X_corner.col = col;
                X_corner.row_confidence = false;
                X_corner.col_confidence = true;
                found_margin = true;
            }
        }
    }
    found_margin = false;
    for (int col=0; col<margin_search_x_bound && !found_margin; ++col) {
        for (int row=0; row<margin_search_y_bound && !found_margin; ++row) {
            if (edge_chase(filtered_image, row, col, W_E)) {
                Y_corner.row = row;
                Y_corner.col = col;
                Y_corner.row_confidence = true;
                Y_corner.col_confidence = false;
                found_margin = true;
            }
        }
    }
    // I candidati ottenuti vengono confrontati per determinare l'angolo in alto a sinistra.
    TL_corner.pick_col(X_corner, Y_corner, min);
    TL_corner.pick_row(X_corner, Y_corner, min);

    // Ricerca dell'angolo in alto a destra.
    X_corner.init(filtered_image.size[1] - 1, 0, false, false);
    Y_corner.init(filtered_image.size[1] - 1, 0, false, false);
    found_margin = false;
    for (int row=0; row<margin_search_y_bound && !found_margin; ++row) {
        for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1]-margin_search_x_bound && !found_margin; --col) {
            if (edge_chase(filtered_image, row, col, N_S)) {
                X_corner.row = row;
                X_corner.col = col;
                X_corner.row_confidence = false;
                X_corner.col_confidence = true;
                found_margin = true;
            }
        }
    }
    found_margin = false;
    for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1]-margin_search_x_bound && !found_margin; --col) {
        for (int row=0; row<margin_search_y_bound && !found_margin; ++row) {
            if (edge_chase(filtered_image, row, col, E_W)) {
                Y_corner.row = row;
                Y_corner.col = col;
                Y_corner.row_confidence = true;
                Y_corner.col_confidence = false;
                found_margin = true;
            }
        }
    }
    // Confronto dei candati
    TR_corner.pick_col(X_corner, Y_corner, max);
    TR_corner.pick_row(X_corner, Y_corner, min);

    // Ricerca dell'angolo in basso a sinistra
    X_corner.init(0, filtered_image.size[0] - 1, false, false);
    Y_corner.init(0, filtered_image.size[0] - 1, false, false);
    found_margin = false;
    for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0]-margin_search_y_bound && !found_margin; --row) {
        for (int col=0; col<margin_search_x_bound && !found_margin; ++col) {
            if (edge_chase(filtered_image, row, col, S_N)) {
                X_corner.row = row;
                X_corner.col = col;
                X_corner.row_confidence = false;
                X_corner.col_confidence = true;
                found_margin = true;
            }
        }
    }
    found_margin = false;
    for (int col=0; col<margin_search_x_bound && !found_margin; ++col) {
        for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0]-margin_search_y_bound && !found_margin; --row) {
            if (edge_chase(filtered_image, row, col, W_E)) {
                Y_corner.row = row;
                Y_corner.col = col;
                Y_corner.row_confidence = true;
                Y_corner.col_confidence = false;
                found_margin = true;
            }
        }
    }
    // Confronto dei candidati
    BL_corner.pick_col(X_corner, Y_corner, min);
    BL_corner.pick_row(X_corner, Y_corner, max);

    // Ricerca dell'angolo in basso a destra
    X_corner.init(filtered_image.size[1] - 1, filtered_image.size[0] - 1, false, false);
    Y_corner.init(filtered_image.size[1] - 1, filtered_image.size[0] - 1, false, false);
    found_margin = false;
    for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0]-margin_search_y_bound && !found_margin; --row) {
        for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1]-margin_search_x_bound && !found_margin; --col) {
            if (edge_chase(filtered_image, row, col, S_N)) {
                X_corner.row = row;
                X_corner.col = col;
                X_corner.row_confidence = false;
                X_corner.col_confidence = true;
                found_margin = true;
            }
        }
    }
    found_margin = false;
    for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1]-margin_search_x_bound && !found_margin; --col) {
        for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0]-margin_search_y_bound && !found_margin; --row) {
            if (edge_chase(filtered_image, row, col, E_W)) {
                Y_corner.row = row;
                Y_corner.col = col;
                Y_corner.row_confidence = true;
                Y_corner.col_confidence = false;
                found_margin = true;
            }
        }
    }
    // Confronto dei candidati
    BR_corner.pick_col(X_corner, Y_corner, max);
    BR_corner.pick_row(X_corner, Y_corner, max);

    // I 4 angoli ottenuti vengono confrontati per costruire un rettangolo che li contenga tutti e 4.
    TL_corner.pick_col(TL_corner, BL_corner, min);
    BL_corner.pick_col(BR_corner, TR_corner, max);
    TL_corner.pick_row(TL_corner, TR_corner, min);
    BL_corner.pick_row(BR_corner, BL_corner, max);

    // Il rettangolo che racchiude il foglio da scannerizzare.
    int height = BR_corner.row - TL_corner.row;
    int width = BR_corner.col - TL_corner.col;
    return {TL_corner.col, TL_corner.row, width, height};
}

/*
 Questa funzione contiene il codice che ricerca le linee bianche che hanno inizio in un pixel candidato per essere un angolo 
 dell'immagine. Il sistema è implementato come una macchina a stati.
*/

bool edge_chase(const Mat &image, int row, int col, int chase_direction) {
    // next_pixel è la funzione utilizzata per muoversi all'interno dell'immagine secondo la direzione dettata dal parametro
    // chase_direction. Possibili direzioni sono Nord -> Sud, Sud -> Nord, Ovest -> Est, Est -> Ovest .
    void (*next_pixel) (int &row, int &col);
    // line_fit è la funzione che determina le coordinate dei pixel che giacciono su una retta descritta dai parametri M, il coefficiente angolare, 
    // start_row e start_col, ovvero il punto da cui ha inizio la retta. Anche in questo caso l'implementazione dipende dalla direzione dettata da
    // chase_direction.
    void (*line_fit) (float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col);

    // L'implementazione di next_pixel e line_fit viene determinata in base a chase_direction
    switch (chase_direction) {
        case W_E:
            next_pixel = next_pixel_W_E;
            line_fit = line_fit_W_E;
            break;
        case E_W:
            next_pixel = next_pixel_E_W;
            line_fit = line_fit_E_W;
            break;
        case N_S:
            next_pixel = next_pixel_N_S;
            line_fit = line_fit_N_S;
            break;
        case S_N:
            next_pixel = next_pixel_S_N;
            line_fit = line_fit_S_N;
            break;
        default:
            std::cerr<<"edge_chasing.edge_chase(): Possible directions are West -> East, East -> West, North -> South, South -> North\n";
            exit(1);
    }

    // Se il pixel di partenza è nero, non può essere un cadidato come angolo, dunque la funzione ritorna falso.
    unsigned char gray_value;
    gray_value = image.at<unsigned char>(row, col);
    if (!gray_value) return false;

    int start_row = row;
    int start_col = col;
    next_pixel(row, col);

    // Vengono inizializzate le variabili di stato
    int iterations = 1;
    int last_shift_iteration = 0;
    int current_state;
    int next_state = KEEP_CHASING;

    // La macchina a stati
    for (;;) {
        current_state = next_state;

        // La macchina a stati inizia nello stato KEEP_CHASING, in cui ricerca una linea di pixel bianchi esattamente orizzontale o esattamente 
        // verticale, in base all'implementazione di next_pixel. 
        // Quando viene trovato un pixel nero, prima di determinare che il pixel di partenza non può essere un angolo, la macchina 
        // a stati effettua una transizione negli stati LOOK_ASIDE_0 e LOOK_ASIDE_1, in cui guarda al valore dei pixel a destra e a sinistra
        // o in alto e in basso (in base al valore del parametro chase_direction). Se uno di questi pixel è bianco vuol dire
        // che la linea attraversata non è perfettamente dritta. Per evitare che vengano inseguite delle rette che hanno tratti molto curvilinei e che quindi
        // sono dei candidati meno validi per essere un margine del foglio, viene imposto all'algoritmo di inseguire i pixel trovati a lato di quello corrente
        // solo se in un numero di iterazioni determinato da un parametro, non sono avvenute altre deviazioni dalla linea retta.
        // Quando intorno al pixel corrente vengono trovati solo pixel neri, oppure quando un pixel laterale bianco non viene inseguito perchè
        // la macchina a stati si trova in un tratto in cui la linea è molto curvilinea, si passa allo stato LOOK_AHEAD. 
        // In questo stato l'algoritmo "guarda avanti" per vedere se dopo pochi pixel neri rincomincia la linea bianca. In tal caso l'inseguimento
        // della riga prosegue, altrimenti si arresta e il pixel di partenza non viene considerato come candidato come angolo della pagina.
        // Se dopo un numero di iterazioni prefissato l'inseguimento non si è arrestato, il pixel di partenza viene considerato come possibile
        // candidato per determinare l'angolo della pagina.
        switch (current_state) {
            case KEEP_CHASING:
                next_pixel(row, col);
                if (!valid_pixel(image, row, col)) return false;
                gray_value = image.at<unsigned char>(row, col);
                
                // Calcolo dello stato futuro
                if (gray_value) {
                    iterations++;
                    if (iterations == PageFrame::CHASE_DEPTH) return true;
                    else next_state = KEEP_CHASING;
                }
                else if (iterations - last_shift_iteration >= PageFrame::ALLOW_SHIFT) {
                    switch (chase_direction) {
                        case W_E:
                        case E_W:
                            if (valid_pixel(image, row, col + 1)) next_state = LOOK_ASIDE_0;
                            else next_state = LOOK_ASIDE_1;
                            break;
                        case N_S:
                        case S_N:
                            if (valid_pixel(image, row + 1, col)) next_state = LOOK_ASIDE_0;
                            else next_state = LOOK_ASIDE_1;
                    }
                }
                else next_state = LOOK_AHEAD;

                break;
            case LOOK_ASIDE_0:
                if (chase_direction == W_E || chase_direction == E_W)
                    gray_value = image.at<unsigned char>(row, col + 1);
                else
                    gray_value = image.at<unsigned char>(row + 1, col);

                // Calcolo dello stato futuro
                if (gray_value) {
                    last_shift_iteration = iterations;
                    iterations++;
                    if (chase_direction == W_E || chase_direction == E_W) ++col;
                    else ++row;

                    if (iterations == PageFrame::CHASE_DEPTH) return true;
                    else next_state = KEEP_CHASING;
                }
                else {
                    switch (chase_direction) {
                        case W_E:
                        case E_W:
                            if (valid_pixel(image, row, col - 1)) next_state = LOOK_ASIDE_1;
                            else next_state = LOOK_AHEAD;
                            break;
                        case N_S:
                        case S_N:
                            if (valid_pixel(image, row - 1, col)) next_state = LOOK_ASIDE_1;
                            else next_state = LOOK_AHEAD;
                    }
                }

                break;
            case LOOK_ASIDE_1:
                if (chase_direction == W_E || chase_direction == E_W)
                    gray_value = image.at<unsigned char>(row, col - 1);
                else
                    gray_value = image.at<unsigned char>(row - 1, col);

                // Calcolo dello stato futuro
                if (gray_value) {
                    last_shift_iteration = iterations;
                    iterations++;
                    if (chase_direction == W_E || chase_direction == E_W) --col;
                    else --row;

                    if (iterations == PageFrame::CHASE_DEPTH) return true;
                    else next_state = KEEP_CHASING;
                }
                else next_state = LOOK_AHEAD;

                break;
            case LOOK_AHEAD:
                // In questo stato l'algoritmo guarda avanti lungo la retta che congiunge il pixel di partenza
                // ed il pixel corrente.
                float M;
                if (chase_direction == W_E || chase_direction == E_W)
                    M = float (row - start_row) / float (col - start_col);
                else M = float (col - start_col) / float (row - start_row);

                int current_row;
                int current_col;
                current_row = row;
                current_col = col;
                int projected_row, projected_col;
                int i;
                for (i=1; i<PageFrame::CHASE_DEPTH/PageFrame::TOLERANCE_FACTOR && !gray_value; ++i) {
                    if (iterations + i >= PageFrame::CHASE_DEPTH) return true;

                    next_pixel(current_row, current_col);
                    line_fit(M, row, col, current_row, current_col, projected_row, projected_col);
                    if (!valid_pixel(image, projected_row, projected_col)) return false;
                    gray_value = image.at<unsigned char>(projected_row, projected_col);
                }

                // Calcolo dello stato futuro
                if (i < PageFrame::CHASE_DEPTH/PageFrame::TOLERANCE_FACTOR || gray_value) {
                    iterations += i;
                    row = projected_row;
                    col = projected_col;
                    next_state = KEEP_CHASING;
                }
                else {
                    return false;
                }

                break;
            default:
                std::cerr<<"edge_chasing.edge_chase(): invalid state code "<<current_state<<"\n";
                exit(1);
        }
    }
}

void next_pixel_W_E(int &row, int &col) {
    col += 1;
}

void next_pixel_E_W(int &row, int &col) {
    col -= 1;
}

void next_pixel_N_S(int &row, int &col) {
    row += 1;
}

void next_pixel_S_N(int &row, int &col) {
    row -= 1;
}

void skip_ahead_W_E(int &row, int &col, int skip) {
    col += skip;
}

void skip_ahead_E_W(int &row, int &col, int skip) {
    col -= skip;
}

void skip_ahead_N_S(int &row, int &col, int skip) {
    row += skip;
}

void skip_ahead_S_N(int &row, int &col, int skip) {
    row -= skip;
}

void line_fit_W_E(float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col) {
    int dx = curr_col - start_col;
    int dy = M*dx;

    projected_row = start_row + dy;
    projected_col = curr_col;
}

void line_fit_E_W(float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col) {
    int dx = start_col - curr_col;
    int dy = (-M)*dx;

    projected_row = start_row + dy;
    projected_col = curr_col;
}

void line_fit_N_S(float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col) {
    int dx = curr_row - start_row;
    int dy = M*dx;

    projected_row = curr_row;
    projected_col = start_col + dy;
}

void line_fit_S_N(float M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col) {
    int dx = start_row - curr_row;
    int dy = (-M)*dx;

    projected_row = curr_row;
    projected_col = start_col + dy;
}

bool valid_pixel(const Mat &image, int row, int col) {
    return row < image.size[0] && col < image.size[1];
}

/*
La seguente funzione è una versione rudimentale di get_page_frame. 
In particolare l'inseguimento delle linee di pixel bianchi consiste semplicemente nella ricerca di linee 
perfettamente dritte (orizzontali o verticali) e senza alcuna interruzione.
Anche la scelta degli angoli tra i candidati ottenuti è semplificata.
*/

Rect rudimentary_get_page_frame(const Mat &filtered_image) {
    int margin_search_x_bound = filtered_image.size[1] / 2;
    int margin_search_y_bound = filtered_image.size[0] / 2;
    int TL_corner[2], TR_corner[2], BL_corner[2], BR_corner[2];

    // Top Left corner search
    int X_stop[2] = {0, 0};
    bool found_margin = false;
    for (int row=0; row<margin_search_y_bound+PageFrame::RUDIMENTARY_DEPTH && !found_margin; ++row) {
        for (int col=0; col<margin_search_x_bound && !found_margin; ++col) {
            boundary = 255;
            for (int k=row; k<row+PageFrame::RUDIMENTARY_DEPTH && boundary; ++k) {
                boundary &= filtered_image.at<unsigned char>(k, col);
            }
            if (boundary) {
                X_stop[0] = row;
                X_stop[1] = col;
                found_margin = true;
            }
        }
    }
    found_margin = false;
    int Y_stop[2] = {0, 0};
    for (int col=0; col<margin_search_x_bound+PageFrame::RUDIMENTARY_DEPTH && !found_margin; ++col) {
        for (int row=0; row<margin_search_y_bound && !found_margin; ++row) {
            boundary = 255;
            for (int k=col; k<col+PageFrame::RUDIMENTARY_DEPTH && boundary; ++k) {
                boundary &= filtered_image.at<unsigned char>(row, k);
            }
            if (boundary) {
                Y_stop[0] = row;
                Y_stop[1] = col;
                found_margin = true;
            }
        }
    }
    TL_corner[0] = min(X_stop[0], Y_stop[0]);
    TL_corner[1] = min(X_stop[1], Y_stop[1]);

    // Top right corner search
    X_stop[0] = 0; X_stop[1] = filtered_image.size[1] - 1;
    Y_stop[0] = 0; Y_stop[1] = filtered_image.size[1] - 1;
    found_margin = false;
    for (int row=0; row<margin_search_y_bound+PageFrame::RUDIMENTARY_DEPTH && !found_margin; ++row) {
        for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1] - margin_search_x_bound - PageFrame::RUDIMENTARY_DEPTH && !found_margin; --col) {
            boundary = 255;
            for (int k=row; k<row+PageFrame::RUDIMENTARY_DEPTH && boundary; ++k) {
                boundary &= filtered_image.at<unsigned char>(k, col);
            }
            if (boundary) {
                X_stop[0] = row;
                X_stop[1] = col;
                found_margin = true;
            }
        }
    }
    found_margin = false;
    for (int col= filtered_image.size[1]-1; col>=filtered_image.size[1] - margin_search_x_bound - PageFrame::RUDIMENTARY_DEPTH && !found_margin; --col) {
        for (int row=0; row<margin_search_y_bound+PageFrame::RUDIMENTARY_DEPTH && !found_margin; ++row) {
            boundary = 255;
            for (int k=col; k>col-PageFrame::RUDIMENTARY_DEPTH && boundary; --k) {
                boundary &= filtered_image.at<unsigned char>(row, k);
            }
            if (boundary) {
                Y_stop[0] = row;
                Y_stop[1] = col;
                found_margin = true;
            }
        }
    }
    TR_corner[0] = min(X_stop[0], Y_stop[0]);
    TR_corner[1] = max(X_stop[1], Y_stop[1]);

    // Bottom left corner search
    X_stop[0] = filtered_image.size[0] - 1; X_stop[1] = 0;
    Y_stop[0] = filtered_image.size[0] - 1; Y_stop[1] = 0;
    found_margin = false;
    for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0] - margin_search_y_bound && !found_margin; --row) {
        for (int col=0; col<margin_search_x_bound+PageFrame::RUDIMENTARY_DEPTH && !found_margin; ++col) {
            boundary = 255;
            for (int k=row; k>row-PageFrame::RUDIMENTARY_DEPTH && boundary; --k) {
                boundary &= filtered_image.at<unsigned char>(k, col);
            }
            if (boundary) {
                X_stop[0] = row;
                X_stop[1] = col;
                found_margin = true;
            }
        }
    }
    found_margin = false;
    for (int col=0; col<margin_search_x_bound+PageFrame::RUDIMENTARY_DEPTH && !found_margin; ++col) {
        for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0] - margin_search_y_bound - PageFrame::RUDIMENTARY_DEPTH && !found_margin; --row) {
            boundary = 255;
            for (int k=col; k<col + PageFrame::RUDIMENTARY_DEPTH && boundary; ++k) {
                boundary &= filtered_image.at<unsigned char>(row, k);
            }
            if (boundary) {
                Y_stop[0] = row;
                Y_stop[1] = col;
                found_margin = true;
            }
        }
    }
    BL_corner[0] = max(X_stop[0], Y_stop[0]);
    BL_corner[1] = min(X_stop[1], Y_stop[1]);

    // Bottom right corner search
    X_stop[0] = filtered_image.size[0] - 1; X_stop[1] = filtered_image.size[1] - 1;
    Y_stop[0] = filtered_image.size[0] - 1; Y_stop[1] = filtered_image.size[1] - 1;
    found_margin = false;
    for (int row=filtered_image.size[0] - 1; row>=filtered_image.size[0] - margin_search_y_bound - PageFrame::RUDIMENTARY_DEPTH && !found_margin; --row) {
        for (int col=filtered_image.size[1] - 1; col>=filtered_image.size[1] - margin_search_x_bound - PageFrame::RUDIMENTARY_DEPTH && !found_margin; --col) {
            boundary = 255;
            for (int k=row; k>row - PageFrame::RUDIMENTARY_DEPTH && boundary; --k) {
                boundary &= filtered_image.at<unsigned char>(k, col);
            }
            if (boundary) {
                X_stop[0] = row;
                X_stop[1] = col;
                found_margin = true;
            }
        }
    }
    found_margin = false;
    for(int col=filtered_image.size[1] - 1; col >= filtered_image.size[1] - margin_search_x_bound - PageFrame::RUDIMENTARY_DEPTH && !found_margin; --col) {
        for (int row=filtered_image.size[0] - 1; row >= filtered_image.size[1] - margin_search_y_bound - PageFrame::RUDIMENTARY_DEPTH && !found_margin; --row) {
            boundary = 255;
            for (int k=col; k>col - PageFrame::RUDIMENTARY_DEPTH && boundary; --k) {
                boundary &= filtered_image.at<unsigned char>(row, k);
            }
            if (boundary) {
                Y_stop[0] = row;
                Y_stop[1] = col;
                found_margin = true;
            }
        }
    }
    BR_corner[0] = max(X_stop[0], Y_stop[0]);
    BR_corner[1] = max(X_stop[1], X_stop[1]);

    // The rectangle that represents the page frame
    int y_offset = min(TL_corner[0], TR_corner[0]);
    int x_offset = min(TL_corner[1], BL_corner[1]);

    int height = max(BL_corner[0], BR_corner[0]) - y_offset - 1;
    int width = max(TR_corner[1], BR_corner[1]) - x_offset - 1;
    return {x_offset, y_offset, width, height};
}

PageFrame::PageFrame(int chase_depth, int tolerance_factor, int allow_shift) {
    CHASE_DEPTH = chase_depth;
    TOLERANCE_FACTOR = tolerance_factor;
    ALLOW_SHIFT = allow_shift;
}