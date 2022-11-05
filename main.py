from virs import VectModelInformationRetrievalSystem
import pickle

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
        backup = input("Cargar Backup: (si/no) \n-> ")
        if backup == 'si':
            try:
                with open(f'irsystem.backup{dataset}', 'rb') as data:    
                    irsystem = pickle.load(data)

            except FileNotFoundError:
                print("No hay ningun dataset cargado actualmente. Generando nuevamente el sistema...")
                irsystem = VectModelInformationRetrievalSystem(0.3, dataset)
        else:
            irsystem = VectModelInformationRetrievalSystem(0.3, dataset)

        with open(f'irsystem.backup{dataset}', 'wb') as data:    
            pickle.dump(irsystem, data)

        main_exec(irsystem)