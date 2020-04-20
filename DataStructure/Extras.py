# Contains exercises that are not data structures
def reverseString(word):
    if not word:
        raise Exception("Not a word")

    stack = []
    for ch in word:
        stack.append(ch)
    result = ""
    while stack:
        result += stack.pop()
    return result


def isBalanced(statement):
    opens = ('(', '[', '<', '{')
    closes = (')', ']', '>', '}')
    stack = []
    for ch in statement:
        if ch in opens:
            stack.append(ch)
        elif ch in closes:
            if stack and opens.index(stack[-1]) == closes.index(ch):
                stack.pop()
            else:
                return False
    return not stack
