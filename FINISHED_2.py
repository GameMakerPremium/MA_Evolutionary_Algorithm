# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 19:30:05 2022

@author: gabri
"""
from datetime import datetime
import pygame
import sys
from pygame.locals import * #import all modules
import random
import math
import csv
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#Settings
clock = pygame.time.Clock()
pygame.init() #initiates pygame
pygame.display.set_caption("Evolutionärer Algorithmus V1") #giving a name to the window
laenge = 1900
breite = 700
WINDOW_SIZE = (laenge,breite) #create window
screen = pygame.display.set_mode(WINDOW_SIZE,0,32) #initiate the window

zyklen = 2
populationsnum = 1
faltflaechen = 8
gelenke = faltflaechen-1
gelenke_hidden = faltflaechen+1
population = []
population2 = []
startpunkt = (500,300)

obstacles = []
origin = (0,0)
rays = []
collision_nr = 0

g = 9.81
sp_list = []
sp_point = (0,0)
g_vectors = []
y_offset = 0
y_boden = 500

draw_points = []
turning_angles = 0

friction_x = 0
x_offset = 0 #wegen Reibung
sum_laenge = 0
hygroskopizität = []

mutation_list = []


#population nr
pop_nr = 0
population_limit = 100 #Wie gross die Anfangspopulation?
selection_ratio = 0.10 #Dezimal

#Chart informations
fps_=80
plt.ion()
def plot_animation(status):
    try:
        df = pd.read_csv('Fitness.csv', index_col=0)
        y = df.index[1:len(df.index)]
        plt.plot(y,'.')
        plt.axis([0, len(df.index), -0.1, 3])
        plt.xlabel('Nr. Individuum')
        plt.ylabel('Relative Geschwindigkeit')
        plt.draw()
        if status == 1:
            print("saved")
            plt.savefig('Fitness_Results.png')
        plt.pause(0.0001)
        # if i == len(df.index)-1:
        #     print("saved")
        #     plt.savefig('foo.png')
        plt.clf()
    except:
        pass
 

#Calculating number of generations
anzahl = population_limit
generations = 0
while anzahl >= 1:
    generations += 1
    anzahl = round(anzahl*(selection_ratio)**generations)
    if anzahl/2 != round(anzahl/2):
        anzahl += 1
    
generations = generations + 1 #Maximale Anzahl an Generationen bis das schnellste Objekt gefunden wurde
generations = 60
nr_in_generation = population_limit




def print_csv(list_population_nr, nummer):
    data = list_population_nr
    if nummer == 0:
        mode = 'a'
        if pop_nr == 0:
            mode = 'w'
        for i in range(len(data)):
            mode = 'a'
            if pop_nr == 0:
                if i == 0:
                    mode = 'w'
            print("To csv", data[i], mode)
            with open('RESULTS.csv', mode) as file:
                    writer = csv.writer(file)
                    writer.writerow(data[i])
            if i == 2:
                if data[i][0] >= 0:
                    if data[i][0] < 3:
                        with open('FITNESS.csv', mode) as file:
                             writer = csv.writer(file)
                             writer.writerow(data[i])
    else:
        for i in range(len(data)):
            for k in range(len(data[i])):
                if i == 0 and k == 0:
                    mode = 'w'
                else:
                    mode = 'a'
                with open('RESULTS.csv', mode) as file:
                    print(data[i], mode)
                    writer = csv.writer(file)
                    writer.writerow(data[i][k])

def mutation_surface():
    random.seed(datetime.now())
    return(random.randint(-10,10))
def mutation_angle():
    random.seed(datetime.now())
    return(random.randint(-5,5))

    
#########################################
def sel_reco_mut(): 
    global population2, nr_in_generation, mutation_list
    in_file = open("RESULTS.csv", "r")
    reader = csv.reader(in_file)
    mylist = list(reader)
    in_file.close()
    print(mylist)
    data_ex = []
    #print all data from csv as list
    for i in range(len(mylist)):
        if len(mylist[i]) == 0:
            pass
        elif mylist[i][0] == 'ï»¿':
            pass
        else:
            data_ex.append(mylist[i])
    
    
    
    #Selektion anhand von Fitness: beste 25% -> höher bis eine gerade Anzahl gefunden worden ist => Paarung
    population3 = []
    gene = 3
    for i in range(population_limit):
        population3.append(data_ex[i*gene:(i+1)*gene])
        
    for i in range(len(population3)):
        for j in range(len(population3[i])):
            for k in range(len(population3[i][j])):
                try:
                    population3[i][j][k] = int(population3[i][j][k]) 
                except:
                    population3[i][j][k] = float(population3[i][j][k]) 
    
    print(population3)
    population3 = sorted(population3, key=lambda tup: tup[2], reverse=True)
    
    print("sortiert: ", population3)
    #Achtung dass +1 weglassen
    
    selection_nr = int((len(population3))*selection_ratio)
    if selection_nr/2 != round(selection_nr/2):
        selection_nr += 1
    
    new_population = []
    #Algorithmus ist beendet
    if selection_nr == 1:
        print("finished")
        plot_animation(1)
        pygame.quit()
        sys.exit()
    else:
        for i in range(selection_nr):
            new_population.append(population3[i])
    nr_in_generation = selection_nr
    print("")
    print("Selection", new_population)
    
    #new_population = random.sample(new_population, len(new_population)) #Liste beliebig mischen
    
    #Rekombination -> Hälfte der Genen werden ausgetauscht (für Fläche und Gelenke)
    nachwuchs = []
    
    # for i in range(int(len(new_population)/2)):
    #     i_new = i*2 #für jedes 2er-Paar
    #     #Gelenke
    #     half1 = int(len(new_population[0][0])/2)
    #     half2 = len(new_population[0][0])-int(len(new_population[0][0])/2)
    #     # print(half1, half2)
    #     part1 = new_population[i_new][0][0:half1]
    #     part2 = new_population[i_new][0][half1:half1+half2]
    #     part3 = new_population[i_new+1][0][0:half1]
    #     part4 = new_population[i_new+1][0][half1:half1+half2]
    #     # print("")
    #     # print(part1, part2, part3, part4)
    #     new_population[i_new][0] = part1+part4
    #     new_population[i_new+1][0] = part3+part2
        
        
        
    #     #Flächen-Längen
    #     half1 = int(len(new_population[0][1])/2)
    #     half2 = len(new_population[0][1])-int(len(new_population[0][1])/2)
    #     # print("Flächen:",half1, half2)
    #     part1 = new_population[i_new][1][0:half1]
    #     part2 = new_population[i_new][1][half1:half1+half2]
    #     part3 = new_population[i_new+1][1][0:half1]
    #     part4 = new_population[i_new+1][1][half1:half1+half2]
    #     # print("")
    #     # print(part1, part2, part3, part4)
    #     new_population[i_new][1] = part1+part4
    #     new_population[i_new+1][1] = part3+part2
    
    for u in range(len(new_population)): #Überbevölkerung + MUTATION
        kinder = 0
        if new_population[u][2][0] > 0.0:
            kinder = 1 #1 Eltern -> 1 Kinder
        if new_population[u][2][0] > 0.5:
            kinder = 8 #1 Eltern -> 1 Kinder    
        if new_population[u][2][0] > 1.0:
            kinder = 10 #1 Eltern -> 2 Kinder
        if new_population[u][2][0] > 1.5:
            kinder = 12 #2 Eltern -> 5 Kinder
        if new_population[u][2][0] > 3:
            kinder = 0 #ERROR
        if new_population[u][2][0] <= 0.0:
            kinder = 0 #Rückwärts   
        for n_kind in range(kinder): #2 Eltern werden zu 4 Kindern
            surfaces = []
            angles = []
            for k in range(len(new_population[u][0])):
                a = random.randint(-18,18)
                if new_population[u][0][k] + a <= 7:
                    surfaces.append(new_population[u][0][k])
                else:
                    surfaces.append(new_population[u][0][k] + a)
            for z in range(len(new_population[u][1])):
                a = random.randint(-10,10)
                if new_population[u][1][z] + a <= -90 or new_population[u][1][z] + a >= 90:
                    angles.append(new_population[u][1][z])
                else:
                    angles.append(new_population[u][1][z] + a)
            new_population[u] = [surfaces,angles,new_population[u][2]]
            nachwuchs.append(new_population[u])
    
      
    # print("")
    # print("Recombination",new_population)
    new_population = []
    new_population = nachwuchs #die Filialgeneration übernimmt, die Parentalgeneration stirbt
    new_population = random.sample(new_population, len(new_population)) #Liste beliebig mischen  
    mutation_list = []
    print("#################")
    print(new_population)
    print("")
    # Mutation: Flächen: +/- 5 , Hygroskopizität: +/- 10
    
    new_2_population = []
    #Error bei Fitness-Berechnung?
    for i in range(len(new_population)):
        if new_population[i][2][0] > 3.0:
            
            pass
        elif new_population[i][2][0] < 0.0:
            
            pass
        else:
            new_2_population.append(new_population[i])
    nr_in_generation = len(new_2_population)
            #data = [[new_population[i][2][0]]]
            # with open('FITNESS.csv', 'a') as file:
            #         writer = csv.writer(file)
            #         writer.writerows(data)
    print_csv(new_2_population,1)
    print("Print to csv:", new_2_population)
    population2 = new_2_population
    
    
#########################################
def humidity_level(frame):
    i = (frame-5) * 0.05
    return(round((math.sin(0.5*i-14)+1)*50,1))#

def startpopulation(populationsnum, o):
    global population, sum_laenge, hygroskopizität, population2
    if generation_nr == 1:
        for i in range(0,populationsnum):
            gelenke_pos=[]
            laengen=[]
            gelenke_hygr=[]
            gelenke_status = []
            for l in range(0,faltflaechen):
                laenge = random.randint(7,100)
                laengen.append(laenge)
                sum_laenge += laenge
            for k in range(0,gelenke_hidden):
                num = random.randint(-90,90)
                gelenke_hygr.append(num)
                hygroskopizität.append(num)
                gelenke_status.append((1,0))
            genotyp = (0, laengen, gelenke_hygr, gelenke_pos, gelenke_status, 0)
            population.append(genotyp)
    else:
            
            gelenke_pos=[]
            laengen=population2[o][0]
            gelenke_hygr=population2[o][1]
            gelenke_status = []
            for l in range(len(population2[o][0])):
                sum_laenge += population2[o][0][l]
            for k in range(len(population2[o][1])):
                hygroskopizität.append(population2[o][1][k])
                gelenke_status.append((1,0))
            genotyp = (0, laengen, gelenke_hygr, gelenke_pos, gelenke_status, 0)
            population.append(genotyp)
            print("Von population2 übernommen:", population)


def collision(population_nr, origin, endpoint, flaeche_nr, frame):
    global collision_nr
    global population
    obstacles = population[population_nr][3]
    obstacle_nr = len(obstacles)-1 #1. weil die Anzahl der Paare 1 kleiner als die Anzahl Flächen ist + (2.) weil die geraden eben gemachte Fläche nicht zählt
    
    #for all starts and ends of all objects n
    for n in range(0,obstacle_nr):
        start = obstacles[n]
        end = obstacles[n+1]
        pygame.draw.line(screen, (255, 0, 0), start, end) #obstacle
        pos = intersect_line_line(start, end, origin, endpoint)
        if pos:
            counter = 0
            #benachbarte Flächen ==> dann war es doch keine Berührung
            if abs(n-flaeche_nr) == 1:
                pass
            else:
                #wirkliche Berührung
                #collision_nr = collision_nr + 1
                #print("collision between:",n,flaeche_nr)
                pygame.draw.circle(screen, (0, 255, 0), (round(pos[0]), round(pos[1])), 3)
                #Gelenke zwischen den Flächen ausschalten
                if n < flaeche_nr:
                    for t in range(n, flaeche_nr):
                        #Winkel schon blockiert? Dann lassen.
                        if population[population_nr][5][t][0] == 0:
                            pass
                        else:
                            population[population_nr][5][t] = (0, frame)
                else:
                    for t in range(flaeche_nr, n):
                        if population[population_nr][5][t][0] == 0:
                            pass
                        else:
                            population[population_nr][5][t] = (0, frame)
                #print(population[population_nr][5])



def intersect_line_line(P0, P1, Q0, Q1):  #P ist obstacle, Q ist ray (bewegliche Linie)
    d = (P1[0]-P0[0]) * (Q1[1]-Q0[1]) + (P1[1]-P0[1]) * (Q0[0]-Q1[0]) 
    #delta x von P * delta y von Q +delta y von P * delta x von Q
    if d == 0:
        return None
    t = ((Q0[0]-P0[0]) * (Q1[1]-Q0[1]) + (Q0[1]-P0[1]) * (Q0[0]-Q1[0])) / d
    #delta zwischen x von P/Q * delta y von Q + delta zwischen y von P/Q * delta x von Q
    u = ((Q0[0]-P0[0]) * (P1[1]-P0[1]) + (Q0[1]-P0[1]) * (P0[0]-P1[0])) / d
    #delta zwischen x von P/Q * delta y von P + delta zwischen y von P/Q * delta x von P
    if 0 <= t <= 1 and 0 <= u <= 1:
        return P1[0] * t + P0[0] * (1-t), P1[1] * t + P0[1] * (1-t)
    return None

def sp_calc(p1, p2, k, population_nr, frame):
    y_coord = p1[1] + (p2[1]-p1[1])/2
    x_coord = p1[0] + (p2[0]-p1[0])/2 
    #drawing SP
    pygame.draw.circle(screen, (255, 0, 0), (x_coord, y_coord), 3)
    
    #calculating "mass"
    
    m = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)*3
    l = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    if len(sp_list) == faltflaechen:
        sp_list[k] = (x_coord, y_coord,l)
    else:
        sp_list.append((x_coord, y_coord,l))
    sp()
    gravity_vec(x_coord, y_coord, m,k, population_nr, frame)
 
def sp():
    global sp_point
    x = 0
    y = 0
    l = 0
    for i in range(len(sp_list)):
        x += sp_list[i][0]*sp_list[i][2]
        y += sp_list[i][1]*sp_list[i][2]
        l += sp_list[i][2]
    try:
        pygame.draw.circle(screen, (0, 0, 255), (x/l,y/l), 5)
        sp_point = (x/l,y/l)
    except:
        print("Error: DIVISION BY 0")
    
 
def gravity_vec(x,y, m,k, population_nr, frame):
    force_magnifier = 0.02
    force = g * force_magnifier * m
    #Einzelne Vektoren zeichnen
    #g_vector = pygame.draw.line(screen, (255,0,0), (x,y),(x,y+force),1)
    if len(g_vectors) == faltflaechen:
        g_vectors[k] = (x,y,force)
        gravity_sum_vec(population_nr, frame)
    else:
        g_vectors.append((x,y,force))

def gravity_sum_vec(population_nr, frame):
    x_sum = 0
    y_sum = 0
    forces_sum = 0
    for i in range(len(g_vectors)):
        x_sum += g_vectors[i][0]
        y_sum += g_vectors[i][1]
        forces_sum += g_vectors[i][2]
    x_sum_avg = x_sum/len(g_vectors) #durchschnittliche x-Koordinate der Vektoren (Angriffspunkte)
    y_sum_avg = y_sum/len(g_vectors)
    forces_sum_avg = forces_sum
    #g_sum_vector = pygame.draw.line(screen, (255,0,0), (x_sum_avg,y_sum_avg),(x_sum_avg,y_sum_avg+forces_sum_avg),1)
    #pygame.draw.circle(screen, (0, 0, 255), (x_sum_avg,y_sum_avg), 3)
    

def gravity(population_nr, frame):
    global y_offset, population, startpunkt, turning_angles, friction_x, x_offset
 
    
    m, p1, p2, z1, z2 = gravity_line(population_nr)
    pygame.draw.circle(screen, (0, 0, 255), p2, 3)
    pygame.draw.circle(screen, (0, 0, 255), p1, 3)
    

    objekt_nr = population_nr
    #ganzes Objekt am "geerdeten" Punkt um einen Winkel drehen
    alpha = math.degrees(math.atan(m))
    alpha = math.radians(alpha)
    turning_angles = alpha
    ################################################
    
    #print("vorher:", frame, population[population_nr][3])
    if m > 0: #EIGENTLICH: m<0 !!!
        #print("m = POSITIV")
        if type(friction_x) == int:
            friction_x = (p1[0], z1)
        elif friction_x[1] == z1:
            x_offset = friction_x[0]-p1[0]
        else:
            friction_x = (p1[0], z1)
        
        draw_points[z1]=p1
        
        #REAL Y_OFFSET
        y_min = p1[1]
        if y_min < y_boden:
            y_offset = abs(y_boden-y_min)
        else:
            y_offset = -1*abs(y_boden-y_min)
        
        #Objekt auf den boden verschieben
        startpunkt = (startpunkt[0], startpunkt[1]+y_offset)
        for u in range(len(population[population_nr][3])):
            population[population_nr][3][u] = (population[population_nr][3][u][0],population[population_nr][3][u][1]+y_offset)
            
                  
        for t in range(len(population[objekt_nr][3])):
                    #distanz zwischen P1 und dem anderen Punkt Pn
                    r = math.sqrt((population[objekt_nr][3][t][0]-p1[0])**2+(population[objekt_nr][3][t][1]-p1[1])**2)
                    if r == 0:
                        pass
                    else:
                        #Drehung um alpha
                        
                        #x = 1*population[objekt_nr][3][t][0]*math.cos(alpha) - population[objekt_nr][3][t][1]*math.sin(alpha)
                        #y = 1*population[objekt_nr][3][t][0]*math.sin(alpha) + population[objekt_nr][3][t][1]*math.cos(alpha)
                        x_2 = population[population_nr][3][t][0]
                        y_2 = population[population_nr][3][t][1]
                        x_1 = p1[0]
                        y_1 = p1[1]
                        
                        if x_1 == x_2:
                            #Punkt ist das Drehzentrum, Koordinaten bleiben somit gleich
                            
                            pass
                        else:
                            m3 = (y_1-y_2)/(x_2-x_1)
                            if population[population_nr][3][t][0] > p1[0]:
                                beta = math.degrees(math.atan(m3))
                                beta = math.radians(beta)
                                x = p1[0]+r*math.cos(beta-alpha)
                                y = p1[1]-r*math.sin(beta-alpha)
                                
                                #turn(x,y,t, population_nr, frame)
                            else: 
                                beta = math.degrees(math.atan(m3))
                                beta = math.radians(beta)
                                x = p1[0]+r*math.cos(beta-alpha+math.radians(180))
                                y = p1[1]-r*math.sin(beta-alpha+math.radians(180))
                                
                                #turn(x,y,t, population_nr, frame)
                            # c = 300
                            # pygame.draw.line(screen, (255, 0, 255), (p1[0],p1[1]), (p1[0]+c,p1[1]-m*c), 6)
                            if t == 0:
                                pygame.draw.circle(screen, (255,0,0), (x,y), 3)
                            else:
                                 pygame.draw.circle(screen, (255,0,255), (x,y), 3)
                            if t > z1:
                                draw_points[t] = (x,y)
                                
                            elif t == z1:
                                pass
                            else:
                                draw_points[t] = (x,y) 
                                
                                  
    else:
        #print("m = NEGATIV")
        if type(friction_x) == int:
            friction_x = (p2[0], z2)
        elif friction_x[1] == z2:
            x_offset = friction_x[0]-p2[0]
        else:
            friction_x = (p2[0], z2)
        draw_points[z2]=p2
        
        #REAL Y_OFFSET
        y_min = p2[1]
        if y_min < y_boden:
            y_offset = abs(y_boden-y_min)
        else:
            y_offset = -1*abs(y_boden-y_min)
        
        #Objekt auf den boden verschieben
        startpunkt = (startpunkt[0], startpunkt[1]+y_offset)
        for u in range(len(population[population_nr][3])):
            population[population_nr][3][u] = (population[population_nr][3][u][0],population[population_nr][3][u][1]+y_offset)
           
        for t in range(len(population[objekt_nr][3])):
                    
                    #distanz zwischen P1 und dem anderen Punkt Pn
                    r = math.sqrt((population[objekt_nr][3][t][0]-p2[0])**2+(population[objekt_nr][3][t][1]-p2[1])**2)
                    if r == 0:
                        pass
                    else:
                        #Drehung um alpha
                        x_2 = population[population_nr][3][t][0]
                        y_2 = population[population_nr][3][t][1]
                        x_1 = p2[0]
                        y_1 = p2[1]
                        if x_1 == x_2:
                            #Punkt ist das Drehzentrum, Koordinaten bleiben somit gleich
                            
                            pass
                        else:
                            m3 = (y_1-y_2)/(x_2-x_1)
                            if population[population_nr][3][t][0] > p2[0]:
                                beta = math.degrees(math.atan(m3))
                                beta = math.radians(beta)
                                x = p2[0]+r*math.cos(beta-alpha)
                                y = p2[1]-r*math.sin(beta-alpha)
                                #turn(x,y,t, population_nr, frame) 
                            else: 
                                
                                beta = math.degrees(math.atan(m3))
                                beta = math.radians(beta)
                                x = p2[0]+r*math.cos(beta-alpha+math.radians(180))
                                y = p2[1]-r*math.sin(beta-alpha+math.radians(180))
                                #turn(x,y,t, population_nr, frame)
                            # c = 300
                            # pygame.draw.line(screen, (255, 0, 255), (p1[0],p1[1]), (p1[0]+c,p1[1]-m*c), 6)
                            if t == 0:
                                pygame.draw.circle(screen, (255,0,0), (x,y), 3)
                            else:
                                pygame.draw.circle(screen, (255,0,255), (x,y), 3)
                            if t > z2:
                                draw_points[t] = (x,y) 
                                
                            elif t == z2:
                                pass
                            else:
                                draw_points[t] = (x,y) 
                                 
                            #print(x,y, r, beta)  
    #print("nachher:", frame, population[population_nr][3])
    #print(frame, draw_points)
    
    
    #population[population_nr] = (population[population_nr][0],population[population_nr][1],population[population_nr][2],draw_points, population[population_nr][4], population[population_nr][5],population[population_nr][6])
        

        
def turn(x,y,n, population_nr, frame):
    global population, startpunkt, draw_points     
    if frame > 10:
        if n == 0:
            population[population_nr][3][n] = (x,y)
        if n == 1:
            population[population_nr][3][n] = (x,y)
    draw_points.append((x,y))   
    pass

    ########################################################

def gravity_line(population_nr):
    global population, sp_point
    Gelenke_Koordinaten_Liste = population[population_nr][3]
    rechts_von_S = []
    links_von_S = []
    a,b = (0,0), (0,0)
    for i in range(len(Gelenke_Koordinaten_Liste)):
        if Gelenke_Koordinaten_Liste[i][0] > sp_point[0]:
            rechts_von_S.append(Gelenke_Koordinaten_Liste[i])
        else:
            links_von_S.append(Gelenke_Koordinaten_Liste[i])
    switch = 1
    while switch == 1:
        #Funktion
        m2 = 0
        for p1 in range(len(links_von_S)):
            for p2 in range(len(rechts_von_S)):
                #m und q bestimmen:
                x1 = links_von_S[p1][0]
                y1 = links_von_S[p1][1]
                x2 = rechts_von_S[p2][0]
                y2 = rechts_von_S[p2][1]
                    
                m = (y2-y1)/(x2-x1)
                m2 = (y1-y2)/(x2-x1)
                q = y1 - m*x1
                
                counter = 0
            
                #falls kein Punkt unter der Funktion -> fertig -> break?
                for z in range(len(Gelenke_Koordinaten_Liste)):
                    x3 = Gelenke_Koordinaten_Liste[z][0]
                    y3 = Gelenke_Koordinaten_Liste[z][1]
                    #Wenn ein Punkt unter der Gerade ist:
                    if  round(y3,1) > round((m*x3 + q),1):
                        #print("con:",round(y3,1),round((m*x3 + q),1), round((m*x3 + q),1)-round(y3,1))
                        #weitersuchen
                        pass
                    else:
                        #Funktion (m, q) und Gelenke (p1, p2) gefunden: 
                        counter += 1
                        pass
                
                if counter == len(Gelenke_Koordinaten_Liste):
                    #Gelenke, die den Boden berühren
                    pygame.draw.line(screen, (255,0,0), (links_von_S[p1][0],links_von_S[p1][1]),(rechts_von_S[p2][0],rechts_von_S[p2][1]),1)
                    a = (links_von_S[p1][0],links_von_S[p1][1])
                    b = (rechts_von_S[p2][0],rechts_von_S[p2][1])
                    #gravity((links_von_S[p1][0],links_von_S[p1][1]),(rechts_von_S[p2][0],rechts_von_S[p2][1]),m, population_nr, frame_nr)
                    #Welche Gelenk-Nummer in der Gelenk-Liste
                    z1, z2 = p1, len(links_von_S)+p2
                    return m2, a, b, z1, z2
                    switch = 0
                else:
                    pass
    

def boden(laenge, breite):
    pygame.draw.line(screen, (0,0,0), (0, y_boden),(laenge,y_boden),3)

def draw_gelenke(x,y):
     pygame.draw.circle(screen, (0, 0, 0), (x,y), 4)
    

def draw(population_nr, frame):
    global population, draw_points, turning_angles, x_offset
    gelenke_pos = []
    gelenke_collision = []
    gelenke_winkel = []
    
    if frame == 1:
        gelenke_pos.append(startpunkt)
        status = population[population_nr][4]
    else:
        gelenke_pos.append(population[population_nr][3][0])
        status = population[population_nr][5]
    
    
    #Für alle Flächen eines Objektes
    for k in range(len(population[population_nr][1])): 
            #Für die erste Fläche
            
                #Biegung der Gelenke? => Blockiert oder nicht?
                if status[k-1][0] == 1:
                    alpha = population[population_nr][2][k-1]*(humidity_level(frame)/100)
                else: 
                    #Winkel blockiert!
                    frame2 = status[k-1][1]
                    #Winkel-Blockierung aufheben?
                    num = int(frame / 252)
                    if 252-(frame2-num*252) < frame-num*252:
                        alpha = population[population_nr][2][k-1]*(humidity_level(frame)/100)
                        status[k-1] = (1, 0)
                    else:
                        #immer noch blockiert
                        alpha = population[population_nr][2][k-1]*(humidity_level(frame2)/100)
                    
                P1x = gelenke_pos[-1][0]
                P1y = gelenke_pos[-1][1]
                
                
                P2x = math.cos(math.radians(alpha))*population[population_nr][1][k] #-beta
                P2y = math.sin(math.radians(alpha))*population[population_nr][1][k] #-beta
                #Ort des Punkt n wird durch vorherigen Punkt n-1 berechnet und zur Gelenke-Liste hinzugefügt
                gelenke_pos.append((P1x+P2x,P1y-P2y))
                #Gelenk und Fläche wird berechnet
                draw_gelenke(P1x+P2x,P1y-P2y)
                pygame.draw.line(screen, (0,0,0), (P1x,P1y),(P1x+P2x,P1y-P2y),3) #surface, color, p1, p2, width
                
                #letzte Fläche hinzugefügt => fertig
                if k+1 == faltflaechen:
                    fitness = 0
                    if frame > 6:
                        fitness_evaluation = []
                        for o in range(len(population[population_nr][3])):
                            fitness_evaluation.append(population[population_nr][3][o][0])
                        
                        fitness = round((max(fitness_evaluation)-startpunkt[0]-sum_laenge)/frame,3)
                        if population[population_nr][6] > fitness:
                            fitness = population[population_nr][6]
                        else:
                            pass
                    population[population_nr] = (population[population_nr][0],population[population_nr][1],population[population_nr][2],gelenke_pos, gelenke_collision, status,fitness) 
                    if frame == zyklen*252-2:
                        print("Print to csv:",population[population_nr][1])
                        print("UND:", [population[population_nr][6]])
                        fitness2 = [population[population_nr][6]]
                        print_csv([population[population_nr][1],hygroskopizität,fitness2],0)
                        
    #checking for collision
    for k in range(len(population[population_nr][3])-1):
        x1 = population[population_nr][3][k][0]
        y1 = population[population_nr][3][k][1]
        x2 = population[population_nr][3][k+1][0]
        y2 = population[population_nr][3][k+1][1]
        collision(population_nr, (x1,y1),(x2,y2), k, frame)
        #SP berechnen
        sp_calc((x1,y1),(x2,y2),k, population_nr, frame)
    
    if frame > 5:
         
        draw_points=[]
        for i in range(faltflaechen+1):
             draw_points.append(" ")
        gravity(population_nr, frame)
        try:
            for i in range(len(draw_points)):
                population[population_nr][3][i] = (draw_points[i][0]+x_offset, draw_points[i][1])
                population[population_nr][2][i] = population[population_nr][2][i]-math.degrees(turning_angles)
        except:
            pass
        #print(population[population_nr])

##################################
#LOOP
generation_nr = 0
print(generations)
while generation_nr < generations:
    generation_nr += 1
    if generation_nr == 1:
        population_limit = population_limit
    else:    
        population_limit = nr_in_generation
    for i in range(population_limit):
        print("Generation",generation_nr, "Objekt", i)
        frame = 0
        population = []
        hygroskopizität = []
        sum_laenge = 0
        while True: #game loop
            screen.fill((255,255,255))
            frame += 1
            
            #Simulation finished?
            if frame > zyklen*252:
                
                pop_nr = pop_nr + 1
                plot_animation(0)
                break
            # if pop_nr >= population_limit:    
            #     pygame.quit()
            #     sys.exit()
            
            #X pressed?
            for event in pygame.event.get():
                if event.type == QUIT: #if X is clicked on the window
                    pygame.quit() #stops pygame
                    sys.exit() #stops the hole code
            
            
            #beginnning of simulation
            if frame == 1:
                startpopulation(populationsnum, i)
                print("Startpopulation")
                print(population)
            
            boden(laenge, breite)    
            #Für jedes Populationsmitglied
            for j in range(0,len(population)):
                    draw(j, frame)           
                    
            
                 
            pygame.display.update()
            clock.tick(fps_) #frame rate : 60 fps
    
    population2 = []
    population = []
    sel_reco_mut()
    if generation_nr+1 == generations:
        plot_animation(1)
    print("KONTROLLE:", population2)
    #print(population2)
    #mit Exemplare von Population2 weitermachen!!!