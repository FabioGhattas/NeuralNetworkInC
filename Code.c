#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <mkl.h>


#define INPUT 784
#define LAYER1 100
#define LAYER2 50
#define OUTPUT 10
#define SAMPLE 100
//Insert here the path to the training images
#define TRAIN_IMAGE "\\MNIST_train.txt"

#define SIZE 784 // 28*28
#define NUM_TRAIN 60000
#define NUM_TEST 10000

unsigned char train_image_char[NUM_TRAIN][SIZE];
unsigned char test_image_char[NUM_TEST][SIZE];
unsigned char train_label_char[NUM_TRAIN][1];
unsigned char test_label_char[NUM_TEST][1];

double train_image[NUM_TRAIN][SIZE];
double test_image[NUM_TEST][SIZE];
int  train_label[NUM_TRAIN];
int test_label[NUM_TEST];
double input[INPUT];
double weights1[LAYER1 * INPUT];
double GradW1[LAYER1 * INPUT];
double biases1[LAYER1];
double GradB1[LAYER1];
double weights2[LAYER2 * LAYER1];
double GradW2[LAYER2 * LAYER1];
double biases2[LAYER2];
double GradB2[LAYER2];
double weights3[OUTPUT * LAYER2];
double GradW3[OUTPUT * LAYER2];
double biases3[OUTPUT];
double GradB3[OUTPUT];
double output1[LAYER1];
double nonSigma1[LAYER1];
double output2[LAYER2];
double nonSigma2[LAYER2];
double output[OUTPUT];
double nonSigma[OUTPUT];


double Derivative(double x);
double MaxIndex(double array[OUTPUT]);
void read_minst();
char* GetLastCh(char* str);

int main() {
    //Per ciascun layer, eccetto l'input, ho i weights, biases e output
    srand(time(NULL));
    //load_mnist();
    //Declarazione variabili
    
    double accuracy = 0;
    //double error = 0;
    double LEARN = 60000 * 500;

    //Settare il result
    double result[OUTPUT];

    //settare a caso i parametri



    for (int i = 0; i < LAYER1; i++) {
        for (int j = 0; j < INPUT; j++) {
            weights1[i * INPUT + j] = ((rand() % 100) / 50.0) - 1;
        }
    }


    for (int i = 0; i < LAYER2; i++) {
        for (int j = 0; j < LAYER1; j++) {
            weights2[i*LAYER1 + j] = ((rand() % 100) / 50.0) - 1;
        }
    }


    for (int i = 0; i < OUTPUT; i++) {
        for (int j = 0; j < LAYER2; j++) {
            weights3[i*LAYER2 + j] = ((rand() % 100) / 50.0) - 1;
        }
    }

    for (int i = 0; i < LAYER1; i++) {
        biases1[i] = ((rand() % 100) / 50.0) - 1;
    }

    for (int i = 0; i < LAYER2; i++) {
        biases2[i] = ((rand() % 100) / 50.0) - 1;
    }

    for (int i = 0; i < OUTPUT; i++) {
        biases3[i] = ((rand() % 100) / 50.0) - 1;
    }

    for (int i = 0; i < LAYER1; i++) {
        for (int j = 0; j < INPUT; j++) {
            GradW1[i*INPUT + j] = 0;
        }
    }

    for (int i = 0; i < LAYER2; i++) {
        for (int j = 0; j < LAYER1; j++) {
            GradW2[i*LAYER1 + j] = 0;
        }
    }

    for (int i = 0; i < OUTPUT; i++) {
        for (int j = 0; j < LAYER2; j++) {
            GradW3[i*LAYER2 + j] = 0;
        }
    }

    for (int i = 0; i < LAYER1; i++) {
        GradB1[i] = 0;
    }

    for (int i = 0; i < LAYER2; i++) {
        GradB2[i] = 0;
    }

    for (int i = 0; i < OUTPUT; i++) {
        GradB3[i] = 0;
    }

    //Inizio i calcoli del neural network
    read_minst();
    LEARN = 5;
    for (int count = 0; count < 60000; count++) {

        if ((count+1) % 1000 == 0) {
            accuracy = accuracy / 10;
            printf("%f\n", accuracy);
            accuracy = 0;
            //LEARN -= 0.03;
        }

        if (count % SAMPLE && count != 0) {
            
            cblas_daxpy(LAYER1 * INPUT, -LEARN / SAMPLE, GradW1, 1, weights1, 1);
            cblas_daxpy(LAYER1 * INPUT, -1, GradW1, 1, GradW1, 1);

            cblas_daxpy(LAYER2 * LAYER1, -LEARN / SAMPLE, GradW2, 1, weights2, 1);
            cblas_daxpy(LAYER2 * LAYER1, -1, GradW2, 1, GradW2, 1);

            cblas_daxpy(OUTPUT * LAYER2, -LEARN / SAMPLE, GradW3, 1, weights3, 1);
            cblas_daxpy(OUTPUT * LAYER2, -1, GradW3, 1, GradW3, 1);


            cblas_daxpy(LAYER1, -LEARN / SAMPLE, GradB1, 1, biases1, 1);
            cblas_daxpy(LAYER1, -1, GradB1, 1, GradB1, 1);

            cblas_daxpy(LAYER2, -LEARN / SAMPLE, GradB2, 1, biases2, 1);
            cblas_daxpy(LAYER2, -1, GradB2, 1, GradB2, 1);

            cblas_daxpy(OUTPUT, -LEARN / SAMPLE, GradB3, 1, biases3, 1);
            cblas_daxpy(OUTPUT, -1, GradB3, 1, GradB3, 1);
        }


        for (int i = 0; i < OUTPUT; i++) {
            if (i == train_label[count]) {
                result[i] = 1;
            }
            else {
                result[i] = 0;
            }
        }

        cblas_dcopy(INPUT, train_image[count], 1, input, 1);

        
        cblas_dcopy(LAYER1, biases1, 1, nonSigma1, 1);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, LAYER1, INPUT, 1, weights1,
            INPUT, input, 1, 1, nonSigma1, 1);

        for (int i = 0; i < LAYER1; i++) {
            output1[i] = 1 / (1 + exp(-nonSigma1[i]));
            //if (nonSigma1[i] < 0)
            //    output1[i] = 0;
            //else
            //    output1[i] = nonSigma1[i];
        }
        
        cblas_dcopy(LAYER2, biases2, 1, nonSigma2, 1);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, LAYER2, LAYER1, 1, weights2,
            LAYER1, output1, 1, 1, nonSigma2, 1);

        for (int i = 0; i < LAYER2; i++) {
            output2[i] = 1 / (1 + exp(-nonSigma2[i]));
            /*if (nonSigma2[i] < 0)
                output2[i] = 0;
            else
                output2[i] = nonSigma2[i];*/
        }

        cblas_dcopy(OUTPUT, biases3, 1, nonSigma, 1);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, OUTPUT, LAYER2, 1, weights3,
            LAYER2, output2, 1, 1, nonSigma, 1);

        for (int i = 0; i < OUTPUT; i++) {
            output[i] = 1 / (1 + exp(-nonSigma[i]));
            /*if (nonSigma[i] < 0)
                output[i] = 0;
            else
                output[i] = nonSigma[i];*/
        }

        if (MaxIndex(output) == train_label[count]) {
            accuracy += 1;
        }
        //printf("Attempt = %f\n", MaxIndex(output));


        //inizio backpropagation


        /*double zDeriv[OUTPUT];
        double zDeriv2[LAYER2];
        double zDeriv1[LAYER1];
        for (int i = 0; i < OUTPUT; i++) {
            zDeriv[i] = Derivative(nonSigma[i]);
        }
        for (int i = 0; i < LAYER2; i++) {
            zDeriv2[i] = Derivative(nonSigma2[i]);
        }
        for (int i = 0; i < LAYER1; i++) {
            zDeriv1[i] = Derivative(nonSigma1[i]);
        }*/
        //TUTTI INSIEME (MOLTO PIù EFFICIENTE)
        for (int i = 0; i < OUTPUT; i++) {
            double first = 2 * (output[i] - result[i]) * Derivative(nonSigma[i]);
            for (int j = 0; j < LAYER2; j++) {
                double second = first * weights3[i*LAYER2 + j] *
                    Derivative(nonSigma2[j]);
                for (int k = 0; k < LAYER1; k++) {
                    double third = second * weights2[j*LAYER1 + k]
                        * Derivative(nonSigma1[k]);
                    for (int l = 0; l < INPUT; l++) {
                        GradW1[k*INPUT + l] += third * input[l];
                    }
                    GradB1[k] += third;
                    GradW2[j*LAYER1 + k] += second * output1[k];
                }
                GradB2[j] += second;
                GradW3[i*LAYER2 + j] += first * output2[j];
            }
            GradB3[i] += first;
        }
    }

    return 0;
}

double Derivative(double x) {
    return (exp(x) /
        ((1 + exp(x)) * (1 + exp(x))));
    /*if (x < 0)
        return 0;
    else if (x > 0)
        return 1;
    else
        return 0.5;*/
}

double MaxIndex(double array[OUTPUT]) {
    double max = 0;
    int index = 0;
    for (int i = 0; i < OUTPUT; i++) {
        if (array[i] > max) {
            max = array[i];
            index = i;
        }
    }
    return index;
}

void print_mnist_pixel(double data_image[][SIZE], int num_data)
{
    int i, j;
    for (i = 0; i < num_data; i++) {
        printf("image %d/%d\n", i + 1, num_data);
        for (j = 0; j < SIZE; j++) {
            printf("%1.1f ", data_image[i][j]);
            if ((j + 1) % 28 == 0) putchar('\n');
        }
        putchar('\n');
    }
}


void print_mnist_label(int data_label[], int num_data)
{
    int i;
    if (num_data == NUM_TRAIN)
        for (i = 0; i < num_data; i++)
            printf("train_label[%d]: %d\n", i, train_label[i]);
    else
        for (i = 0; i < num_data; i++)
            printf("test_label[%d]: %d\n", i, test_label[i]);
}


void read_minst() {
    FILE* ptr = NULL;
    char ch;

    // Opening file in reading mode
    ptr = fopen(TRAIN_IMAGE, "r");

    if (NULL == ptr) {
        printf("file can't be opened\n");
    }
    char str[5] = "";
    printf("content of this file are \n");
    /*fgets(str, 3000, ptr);
    printf(str);
    printf("\n");*/

    // Printing what is written in file
    // character by character using loop.
    int count1 = 0;
    int count2 = 0;

    int commacount = 0;
    do {
        ch = fgetc(ptr);
        if (ch != ',') {
            if (strlen(str) == 0) {
                str[0] = ch;
                str[1] = '\0';
            }
            else {
                char str1[2];
                str1[0] = ch;
                str1[1] = '\0';
                strcat_s(str, 5, str1);
            }
            //printf("mo vediamo0\n");
        }
        else {
            commacount++;
            //printf("mo vediamo,\n");
            if (commacount == 785) {
                train_image[count1][783] = 0;
                count1++;
                count2 = 0;
                train_label[count1] = atoi(GetLastCh(str));
                count2++;
                commacount = 1;
                //printf("mo vediamoriga\n");
            }
            if (count2 == 0) {
                train_label[count1] = atoi(str);
            }
            else {
                train_image[count1][count2 - 1] = atoi(str) / 255.0;
                //printf(str);
                //printf("\n");
            }
            count2++;
            if (count1 > 59999) {
                break;
            }

            str[0] = '\0';
        }

        // Checking if character is not EOF.
        // If it is EOF stop reading.
    } while (ch != EOF);
    printf("Salvato i numeri\n");
    // Closing the file
    fclose(ptr);
}

char* GetLastCh(char* str) {
    char ch = *str;
    int i = 0;
    while (ch != '\0') {
        i++;
        ch = str[i];
    }
    return &str[i - 1];
}