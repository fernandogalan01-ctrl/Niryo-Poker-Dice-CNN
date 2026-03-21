from collections import Counter
def evaluate_poker_hand(predictions):
    # es importante contar las apariciones de cada cara para determinar la mano de póker
    counts = sorted(Counter(predictions).values(), reverse=True)
    # es importante ordenar los counts para comparar con las manos de póker
    unique_faces = sorted(list(set(predictions)))
    if 5 in counts: return 7, "Póker"
    if counts == [3, 2]: return 6, "Full House"
    if len(unique_faces) == 5:
        # comprobar si las caras forman una escalera 
        return 5, "Escalera"
    if 3 in counts: return 4, "Trío"
    if counts == [2, 2, 1]: return 3, "Doble Par"
    if 2 in counts: return 2, "Par"
    return 1, "Nada"