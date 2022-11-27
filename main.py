from virs import VectModelInformationRetrievalSystem
from boolean import BooleanModel

def main_exec(irsystem):
    while True:
        print('\nOpciones:')
        mode = input(f"1 - Hacer consulta \nEnter - Para terminar\n-> ") #The idea is to extend it and add options like system evaluation, etc (adding other modes)
        if mode == '1':
            query = input("\nEscribe la consulta: ")
            alpha = input("Escribe la constante de suavizado: ")
            irsystem.search(query, alpha)
        else:
            break

if __name__ == '__main__':
    dataset = input('Elige un Dataset: \n1 - Cranfield \n2 - MED \nEnter - Para terminar\n-> ')
    if dataset == '1' or dataset == '2':
        irsystem = BooleanModel(0.3,dataset)
        main_exec(irsystem)