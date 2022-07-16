#include "page_frame.h"
#include "opencv2/opencv.hpp"
#include "utility.h"
#include "corners.h"

using namespace cv;

int PageFrame::CHASE_DEPTH = 400;
int PageFrame::RUDIMENTARY_DEPTH = 200;
int PageFrame::MAX_ADJUSTMENTS = 6;
double PageFrame::TANGENT_TABLE[] = {
        -0.268, // tan(-15°)
        -0.176, // tan(-10°)
        -0.087, // tan(-5°)
        0.087, // tan(5°)
        0.176, // tan(10°)
        0.268, // tan(15°)
};

/*
 Questa funzione estrae il rettangolo contenente il foglio da scannerizzare ricercandone i 4 angoli.
 Per esempio, per ricercare l'angolo in alto a sinista, l'immagine viene attraversata partendo dal pixel nella
 posizione (0, 0), scorrendone le colonne di ciascuna riga alla ricerca di un pixel bianco (che corrisponde ad
 un punto posto in risalto dall'operazione precedente). Quando viene trovato un pixel bianco, la funzione
 "edge_chase" determina se il pixel appartiene ad un contorno del foglio cercando di percorrrere una linea
 di pixel bianchi diretta verso il basso, che parta da quel pixel.
 Se una linea di questo tipo viene trovata, il pixel diventa un candidato per l'angolo ricercato.
 Successivamente l'immagine viene nuovamente attraversata partendo dal pixel in (0, 0), questa volta però
 scorredone le righe, alla ricerca di un pixel bianco da cui abbia inizio un linea bianca diretta da sinistra a destra.
 I candidati ottenuti scorrendo le righe e le colonne vengono confrontati tramite delle funzioni apposite del
 modulo corners.h, per determinare la posizione dell'angolo in alto a sinistra.
 Tramite un procedimento analogo vengono ricercati gli angoli in basso a sinistra, in basso a destra ed in alto a destra.
*/

Rect get_page_frame(const Mat &base_image, const Mat &filtered_image) {
    std::vector<std::vector<Point>> contours = std::vector<std::vector<Point>>();
    std::vector<std::vector<Point>> corners = std::vector<std::vector<Point>>();
    Scalar TL_color(0, 255, 255);
    Scalar TR_color(0, 128, 255);
    Scalar BL_color(0, 255, 0);
    Scalar BR_color(0, 0, 255);
    Mat contours_drawing = filtered_image.clone();
    cvtColor(contours_drawing, contours_drawing, COLOR_GRAY2RGB);
    for (int i=0; i<8; ++i) contours.emplace_back();
    for (int i=0; i<4; ++i) corners.emplace_back();

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
            if (edge_chase(filtered_image, row, col, N_S, contours[0])) {
                X_corner.row = row;
                X_corner.col = col;
                X_corner.row_confidence = false;
                X_corner.col_confidence = true;
                found_margin = true;
            }
            else contours[0].clear();
        }
    }
    found_margin = false;
    for (int col=0; col<margin_search_x_bound && !found_margin; ++col) {
        for (int row=0; row<margin_search_y_bound && !found_margin; ++row) {
            if (edge_chase(filtered_image, row, col, W_E, contours[1])) {
                Y_corner.row = row;
                Y_corner.col = col;
                Y_corner.row_confidence = true;
                Y_corner.col_confidence = false;
                found_margin = true;
            }
            else contours[1].clear();
        }
    }
    if (contours[0].empty()) contours[0].emplace_back(0, 0);
    drawContours(contours_drawing, contours, 0, TL_color, 30);
    if (contours[1].empty()) contours[1].emplace_back(0, 0);
    drawContours(contours_drawing, contours, 1, TL_color, 30);
    // I candidati ottenuti vengono confrontati per determinare l'angolo in alto a sinistra.
    TL_corner.pick_col(X_corner, Y_corner, min);
    TL_corner.pick_row(X_corner, Y_corner, min);
    corners[0].emplace_back(TL_corner.col, TL_corner.row);
    drawContours(contours_drawing, corners, 0, TL_color, 100);

    // Ricerca dell'angolo in alto a destra.
    X_corner.init(filtered_image.size[1] - 1, 0, false, false);
    Y_corner.init(filtered_image.size[1] - 1, 0, false, false);
    found_margin = false;
    for (int row=0; row<margin_search_y_bound && !found_margin; ++row) {
        for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1]-margin_search_x_bound && !found_margin; --col) {
            if (edge_chase(filtered_image, row, col, N_S, contours[2])) {
                X_corner.row = row;
                X_corner.col = col;
                X_corner.row_confidence = false;
                X_corner.col_confidence = true;
                found_margin = true;
            }
            else contours[2].clear();
        }
    }
    found_margin = false;
    for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1]-margin_search_x_bound && !found_margin; --col) {
        for (int row=0; row<margin_search_y_bound && !found_margin; ++row) {
            if (edge_chase(filtered_image, row, col, E_W, contours[3])) {
                Y_corner.row = row;
                Y_corner.col = col;
                Y_corner.row_confidence = true;
                Y_corner.col_confidence = false;
                found_margin = true;
            }
            else contours[3].clear();
        }
    }
    if (contours[2].empty()) contours[2].emplace_back(base_image.size[1]-1, 0);
    drawContours(contours_drawing, contours, 2, TR_color, 30);
    if (contours[3].empty()) contours[3].emplace_back(base_image.size[1]-1, 0);
    drawContours(contours_drawing, contours, 3, TR_color, 30);
    // Confronto dei candidati
    TR_corner.pick_col(X_corner, Y_corner, max);
    TR_corner.pick_row(X_corner, Y_corner, min);
    corners[1].emplace_back(TR_corner.col, TR_corner.row);
    drawContours(contours_drawing, corners, 1, TR_color, 100);

    // Ricerca dell'angolo in basso a sinistra
    X_corner.init(0, filtered_image.size[0] - 1, false, false);
    Y_corner.init(0, filtered_image.size[0] - 1, false, false);
    found_margin = false;
    for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0]-margin_search_y_bound && !found_margin; --row) {
        for (int col=0; col<margin_search_x_bound && !found_margin; ++col) {
            if (edge_chase(filtered_image, row, col, S_N, contours[4])) {
                X_corner.row = row;
                X_corner.col = col;
                X_corner.row_confidence = false;
                X_corner.col_confidence = true;
                found_margin = true;
            }
            else contours[4].clear();
        }
    }
    found_margin = false;
    for (int col=0; col<margin_search_x_bound && !found_margin; ++col) {
        for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0]-margin_search_y_bound && !found_margin; --row) {
            if (edge_chase(filtered_image, row, col, W_E, contours[5])) {
                Y_corner.row = row;
                Y_corner.col = col;
                Y_corner.row_confidence = true;
                Y_corner.col_confidence = false;
                found_margin = true;
            }
            else contours[5].clear();
        }
    }
    if (contours[4].empty()) contours[4].emplace_back(0, base_image.size[0]-1);
    drawContours(contours_drawing, contours, 4, BL_color, 30);
    if (contours[5].empty()) contours[5].emplace_back(0, base_image.size[0]-1);
    drawContours(contours_drawing, contours, 5, BL_color, 30);
    // Confronto dei candidati
    BL_corner.pick_col(X_corner, Y_corner, min);
    BL_corner.pick_row(X_corner, Y_corner, max);
    corners[2].emplace_back(BL_corner.col, BL_corner.row);
    drawContours(contours_drawing, corners, 2, BL_color, 100);

    // Ricerca dell'angolo in basso a destra
    X_corner.init(filtered_image.size[1] - 1, filtered_image.size[0] - 1, false, false);
    Y_corner.init(filtered_image.size[1] - 1, filtered_image.size[0] - 1, false, false);
    found_margin = false;
    for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0]-margin_search_y_bound && !found_margin; --row) {
        for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1]-margin_search_x_bound && !found_margin; --col) {
            if (edge_chase(filtered_image, row, col, S_N, contours[6])) {
                X_corner.row = row;
                X_corner.col = col;
                X_corner.row_confidence = false;
                X_corner.col_confidence = true;
                found_margin = true;
            }
            else contours[6].clear();
        }
    }
    found_margin = false;
    for (int col=filtered_image.size[1]-1; col>=filtered_image.size[1]-margin_search_x_bound && !found_margin; --col) {
        for (int row=filtered_image.size[0]-1; row>=filtered_image.size[0]-margin_search_y_bound && !found_margin; --row) {
            if (edge_chase(filtered_image, row, col, E_W, contours[7])) {
                Y_corner.row = row;
                Y_corner.col = col;
                Y_corner.row_confidence = true;
                Y_corner.col_confidence = false;
                found_margin = true;
            }
            else contours[7].clear();
        }
    }
    if (contours[6].empty()) contours[6].emplace_back(base_image.size[1]-1, base_image.size[0]-1);
    drawContours(contours_drawing, contours, 6, BR_color, 30);
    if (contours[7].empty()) contours[7].emplace_back(base_image.size[1]-1, base_image.size[0]-1);
    drawContours(contours_drawing, contours, 7, BR_color, 30);
    // Confronto dei candidati
    BR_corner.pick_col(X_corner, Y_corner, max);
    BR_corner.pick_row(X_corner, Y_corner, max);
    corners[3].emplace_back(BR_corner.col, BR_corner.row);
    drawContours(contours_drawing, corners, 3, BR_color, 100);

    // I 4 angoli ottenuti descrivono un parallelogramma che non necessariamente ha lati perfettamente orizzontali
    // o perfettamente verticali. Dunque gli angoli vengono confrontati per costruire un rettangolo 
    // che li contenga tutti e 4. 
    TL_corner.pick_col(TL_corner, BL_corner, min);
    BR_corner.pick_col(BR_corner, TR_corner, max);
    TL_corner.pick_row(TL_corner, TR_corner, min);
    BR_corner.pick_row(BR_corner, BL_corner, max);

    // Il rettangolo che racchiude il foglio da scannerizzare.
    int height = BR_corner.row - TL_corner.row;
    int width = BR_corner.col - TL_corner.col;
    imshow("contours", contours_drawing);
    return {TL_corner.col, TL_corner.row, width, height};
}

/*
 Questa funzione contiene il codice che ricerca le linee bianche che hanno inizio in un pixel candidato per essere un angolo 
 dell'immagine. Il sistema è implementato come una macchina a stati.
*/

bool edge_chase(const Mat &image, int row, int col, int chase_direction, std::vector<Point> &contour) {
    // next_pixel è la funzione utilizzata per muoversi all'interno dell'immagine secondo la direzione dettata dal
    // parametro chase_direction. Possibili direzioni sono Nord -> Sud, Sud -> Nord, Ovest -> Est, Est -> Ovest .
    void (*next_pixel) (int &row, int &col);
    // line_fit è la funzione che determina le coordinate dei pixel che giacciono su una retta descritta dai parametri 
    // M, il coefficiente angolare, start_row e start_col, ovvero il punto da cui ha inizio la retta. 
    // Anche in questo caso l'implementazione dipende dalla direzione dettata da chase_direction.
    void (*line_fit) (double M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col);

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
    contour.emplace_back(col, row);

    int start_row = row;
    int start_col = col;
    int projected_row, projected_col;
    next_pixel(row, col);

    // Vengono inizializzate le variabili di stato
    int iterations = 1;
    int adjustments = 0;
    double M;
    int current_state;
    int next_state = KEEP_CHASING;

    // La macchina a stati
    for (;;) {
        current_state = next_state;

        // L'automa a stati inizia nello stato KEEP_CHASING, in cui ricerca una linea di pixel bianchi esattamente 
        // orizzontale o esattamente verticale, in base all'implementazione scelta di next_pixel. 
        // Quando viene trovato un pixel nero e se l'inseguimento si è interrotto dopo una sequenza di lunghezza
        // predefinita di avanzamenti che hanno avuto successo, l'automa non conclude immediatamente che il pixel
        // di partenza non può essere un angolo, ma effettua una transizione verso lo stato ADJUST_ORIENTATION.
        // Se l'inseguimento invece si interrompe dopo una breve sequenza di pixel bianchi, il pixel di partenza non 
        // può essere un angolo.
        // Fintanto che si trova nello stato KEEP_CHASING, l'automa cerca di inseguire rette perfettamente orizzontali
        // o perfettamente verticali. Quando entra in ADJUST_ORIENTATION cerca di cambiare l'orientazione della retta da
        // inseguire ed entra nello stato FIT_LINE.
        // La prima volta l'orientazione scelta forma un angolo di -15° rispetto all'orizzontale, se l'inseguimento lungo
        // tale retta fallisce, l'automa prova con un'orientazione pari a -10°, se fallisce di nuovo ritenta con
        // orientazione pari a -5°, quindi 5°, 10°, e 15°. Se di nuovo fallisce l'automa decreta finalmente che
        // il pixel di partenza non può essere un angolo.
        switch (current_state) {
            case KEEP_CHASING:
                next_pixel(row, col);
                if (!valid_pixel(image, row, col)) return false;
                gray_value = image.at<unsigned char>(row, col);
                
                // Calcolo dello stato futuro
                if (gray_value) {
                    iterations++;
                    contour.emplace_back(col, row);
                    if (iterations == PageFrame::CHASE_DEPTH) return true;
                    else next_state = KEEP_CHASING;
                }
                else if (iterations >= PageFrame::CHASE_DEPTH / 2)  next_state = ADJUST_ORIENTATION;
                else return false;

                break;
            case ADJUST_ORIENTATION:
                // Dopo 6 tentativi di aggiustamento dell'angolo, l'automa si arrende.
                if (adjustments == PageFrame::MAX_ADJUSTMENTS) return false;

                // Reset delle variabili di stato
                contour.clear();
                M = PageFrame::TANGENT_TABLE[adjustments++];
                row = start_row;
                col = start_col;
                iterations = 1;

                // Stato futuro
                next_state = FIT_LINE;

                break;
            case FIT_LINE:
                next_pixel(row, col);
                line_fit(M, start_row, start_col, row, col, projected_row, projected_col);
                gray_value = image.at<unsigned char>(projected_row, projected_col);

                // Calcolo dello stato futuro
                if (gray_value) {
                    contour.emplace_back(projected_col, projected_row);
                    iterations++;
                    if (iterations == PageFrame::CHASE_DEPTH) return true;
                    else next_state = FIT_LINE;
                }
                else next_state = ADJUST_ORIENTATION;

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

void line_fit_W_E(double M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col) {
    int dx = curr_col - start_col;
    int dy = M*dx;

    projected_row = start_row + dy;
    projected_col = curr_col;
}

void line_fit_E_W(double M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col) {
    int dx = start_col - curr_col;
    int dy = (-M)*dx;

    projected_row = start_row + dy;
    projected_col = curr_col;
}

void line_fit_N_S(double M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col) {
    int dx = curr_row - start_row;
    int dy = M*dx;

    projected_row = curr_row;
    projected_col = start_col + dy;
}

void line_fit_S_N(double M, int start_row, int start_col, int curr_row, int curr_col, int &projected_row, int &projected_col) {
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
    unsigned char boundary;

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

PageFrame::PageFrame(int chase_depth, int max_adjustments, int rudimentary_depth) {
    CHASE_DEPTH = chase_depth;
    MAX_ADJUSTMENTS = max_adjustments;
    RUDIMENTARY_DEPTH = rudimentary_depth;
}