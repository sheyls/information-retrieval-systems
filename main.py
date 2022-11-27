from virs import VectModelInformationRetrievalSystem

def main_exec(irsystem):
    while True:
        print('\nOpciones:')
        mode = input(f"1 - Hacer consulta \n2 - Aplicar retroalimentación de Rocchio (modelo vectorial) \nEnter - Para terminar\n-> ") #The idea is to extend it and add options like system evaluation, etc (adding other modes)
        if mode == '1':
            query = input("\nEscribe la consulta: ")
            alpha = input("Escribe la constante de suavizado: ")
            irsystem.search(query, alpha)
        elif mode == '2':
            if not isinstance(irsystem, VectModelInformationRetrievalSystem):
                print("\nEL algoritmo de Rocchio está implementado para el modelo vectorial.")
            else:
                print('\nAplicar Retroalimentación de Rocchio a:')
                ask = [f'{query[0]} - {irsystem.queries[query[0]]["text"]}\n' for query in irsystem.searched.items()]
                query_id = input("".join(ask) + 'Elegir ID -> ')

                print("\n---------- Documentos Recuperados -----------\n")

                irsystem.search(query_id = query_id, preview=250)

                relevants = input("Seleccione el ID de los documentos que le parecen relevantes separados por espacios: \n->").split(' ')
                
                irsystem.executeRocchio(query_id, relevants, 1, 0.9, 0.5)

                print("\n---------- Algoritmo de Rocchio realizado correctamente -----------\n")

                irsystem.search(query_id = query_id, preview=250)
                irsystem.evaluate_query(query_id, True)
        
        else:
            break

if __name__ == '__main__':
    dataset = input('Elige un Dataset: \n1 - Cranfield \n2 - MED \nEnter - Para terminar\n-> ')
    if dataset == '1' or dataset == '2':
        irsystem = VectModelInformationRetrievalSystem(0.3, dataset)
        main_exec(irsystem)