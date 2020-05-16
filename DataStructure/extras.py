def odd_even_list_my_own_ds(first):
    if first == None:
        return None

    odd = first
    even = first.next
    evenHead = even

    while even != None and even.next != None:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next

    odd.next = evenHead
    return first
