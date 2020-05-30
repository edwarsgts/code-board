#  May LeetCode Challenge Archives

# Day 1 : First Bad Version


def first_bad_version_0(n):
    # Using Binary Search
    left = 1
    right = n
    while left < right:
        mid = (left + right) // 2
        if isBadVersion(mid):
            right = mid
        else:
            left = mid + 1
    return left


# def first_bad_version_1(n):
#     import bisect
#     # using Bisect
#     self.__getitem__ = isBadVersion
#     return bisect.bisect(self, True, 1, n)


def isBadVersion():
    # function provided by question
    return

# Day 2 : Jewels in Stones


def num_jewels_in_stones(J, S):
    result = 0
    for jewel in J:
        result += S.count(jewel)
    return result

# Day 3: Ransom Note


def can_construct(ransomNote, magazine):
    for ch in ransomNote:
        if ch not in magazine:
            return False
        magazine = magazine.replace(ch, "", 1)
    return True

#  Day 4: Number Complement


def find_complement(num):
    b_num = bin(num)
    res = ''.join(['0' if r == '1' else '1' for r in b_num[2:]])
    return int(res, 2)

# Day 5 First Unique Character in a String


def first_uniq_char(s):
    import collections
    freq = collections.Counter(s)
    for k, v in freq.items():
        if v == 1:
            return s.index(k)
    return -1

# Day 6 : Majority Element


def majority_element(nums):
    # if count of item is > len (num)/2,
    # it is consiered a majority element
    nums_set = set(nums)
    for i in nums_set:
        if nums.count(i) > len(nums)/2:
            return i

# Day 7 : Cousins of a Binary Tree


def is_cousin(root, x, y):
    def depth(root, x, par, level):
        if not root:
            return

        if root.val == x:
            return [level, par]

        return depth(root.left, x, root, level+1) or depth(root.right, x, root, level+1)

        a = depth(root, x, None, 1)
        b = depth(root, y, None, 1)
        if a[0] == b[0] and a[1] != b[1]:
            return True
        return False

# Day 8: Check if it is a straight line


def check_straight_line(coordinates):
    #  To check if it is a straight line, using
    # gradient to check, the expression is:
    # (y - y1) / (x - x1) = (y1 - y0) / (x1 - x0)
    # But to avoid zero division error, use multiplication form:
    # dx * (y - y1) = dy * (x - x1), where dx = x1 - x0 and dy = y1 - y0
    (x0, y0), (x1, y1) = coordinates[: 2]
    for x, y in coordinates:
        if (x1 - x0) * (y - y1) != (x - x1) * (y1 - y0):
            return False
    return True


def check_straight_line_2(coordinates):
    (x0, y0), (x1, y1) = coordinates[:2]
    return all((x1-x0) * (y - y1) == (x-x1) * (y1 - y0) for x, y in coordinates)

# Day 9: Valid Perfect Square


def is_perfect_square(num):
    # Newton's method
    r = num
    while r*r > num:
        r = (r + num/r) // 2
    return r*r == num

# Day 10: Find town Judge


def find_judge(N, trust):
    """
    In a town, there are N people labelled from 1 to N.  There is a rumor that one
    of these people is secretly the town judge.
    If the town judge exists, then:
    1. The town judge trusts nobody.
    2. Everybody (except for the town judge) trusts the town judge.
    3. There is exactly one person that satisfies properties 1 and 2.
    You are given trust, an array of pairs trust[i] = [a, b] representing that the person labelled a trusts the person labelled b.
    If the town judge exists and can be identified,
    return the label of the town judge.  Otherwise, return -1.
    """
    count = [0] * (N+1)
    for i, j in trust:
        count[i] -= 1
        count[j] += 1
    for i in range(1, N+1):
        if count[i] == N - 1:
            return i
    return -1

# Day 11: Flood fill


def flood_fill(image, sr, sc, newColor):
    if image[sr][sc] == newColor:
        return

    original_color = image[sr][sc]

    def helper(image, sr, sc, newColor):
        sr_not_valid = sr < 0 or sr >= len(image)
        sc_not_valid = sc < 0 or sc >= len(image[0])
        # base condition , if out of grid/not original color
        if sr_not_valid or sc_not_valid or image[sr][sc] != original_color:
            return
        image[sr][sc] = newColor
        helper(image, sr-1, sc, newColor)
        helper(image, sr+1, sc, newColor)
        helper(image, sr, sc-1, newColor)
        helper(image, sr, sc+1, newColor)
    helper(image, sr, sc, newColor)
    return image


def flood_fill_best_time(image, sr, sc, newColor):
    if len(image) == 0:
        return []

    X, Y = len(image), len(image[0])
    color = image[sr][sc]

    if color == newColor:
        return image

    frontier_stack = [(sr, sc)]
    while len(frontier_stack) > 0:
        point = frontier_stack.pop()
        x, y = point[0], point[1]
        image[x][y] = newColor

        candidates = ((x + 1, y),
                      (x - 1, y),
                      (x, y + 1),
                      (x, y - 1),)
        for cand in candidates:
            x, y = cand[0], cand[1]
            if x >= X or x < 0 or y >= Y or y < 0:
                continue
            if image[x][y] == color:
                frontier_stack.append(cand)

    return image

# Day 12: Single Element in a Sorted Array


def single_non_duplicate(nums):
    import collections
    # num_freq = collections.OrderedDict(collections.Counter(nums))
    num_freq = collections.Counter(nums)
    for k, v in num_freq.items():
        if v == 1:
            return k


test_input = [1, 1, 2, 3, 3, 4, 4, 8, 8]
expected_output = 2


def single_non_duplicate_60ms(nums):
    """
    1  1  4  4  5  5  6  8  8
               mid .
    1  1  4  5  5  6  6  8  8  9
             .
    """
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if mid < len(nums) - 1 and nums[mid] == nums[mid + 1]:
            halves_are_even = right - mid - 1
            if halves_are_even % 2 == 1:
                left = mid + 2
            else:
                right = mid - 1
        elif mid > 0 and nums[mid] == nums[mid - 1]:
            halves_are_even = right - mid
            if halves_are_even % 2 == 1:
                left = mid + 1
            else:
                right = mid - 2
        else:
            return nums[mid]
    return -1

# Day 13: Remove K Digits


def removeKdigits(num: str, k: int) -> str:
    out = []
    for d in num:
        while k and out and out[-1] > d:
            out.pop()
            k -= 1
        out.append(d)
    return ''.join(out[:-k or None]).lstrip('0') or '0'

# Day 14: Implement Trie


class Trie_116ms:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = {}

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        t = self.trie
        for c in word:
            if c not in t:
                t[c] = {}
            t = t[c]
        # Used to signify end of word
        t['#'] = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        t = self.trie
        for c in word:
            if c not in t:
                return False
            t = t[c]
        return '#' in t

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        t = self.trie
        for c in prefix:
            if c not in t:
                return False
            t = t[c]
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

class Trie:
    # 196ms
    class TrieNode:
        def __init__(self):
            self.word = False
            self.children = {}

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = self.TrieNode()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self.root
        for i in word:
            if i not in node.children:
                node.children[i] = self.TrieNode()
            node = node.children[i]
        node.word = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node = self.root
        for i in word:
            if i not in node.children:
                return False
            node = node.children[i]
        return node.word

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.root
        for i in prefix:
            if i not in node.children:
                return False
            node = node.children[i]
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

# Day 15: Maximum sum circular subarray

def maxSubarraySumCircular(A):
    total, maxSum, curMax, minSum, curMin = 0, - \
        float('inf'), 0, float('inf'), 0
    for a in A:
        curMax = max(curMax + a, a)
        maxSum = max(maxSum, curMax)
        curMin = min(curMin + a, a)
        minSum = min(minSum, curMin)
        total += a
    return max(maxSum, total - minSum) if maxSum > 0 else maxSum

# Day 16 : odd and even nodes in a linkedlist


def oddEvenList(head):
    if not head:
        return None

    odd = head
    even = head.next
    evenHead = even

    while odd.next and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next

    odd.next = evenHead
    return head

# Day 17: Find all Anagrams in a String


def find_anagrams(s, p):
    import collections
    res = []
    pCounter = collections.Counter(p)
    sCounter = collections.Counter(s[:len(p)-1])
    for i in range(len(p)-1, len(s)):
        sCounter[s[i]] += 1   # include a new char in the window
        # This step is O(1), since there are at most 26 English letters
        if sCounter == pCounter:
            res.append(i-len(p)+1)   # append the starting index
        # decrease the count of oldest char in the window
        sCounter[s[i-len(p)+1]] -= 1
        if sCounter[s[i-len(p)+1]] == 0:
            del sCounter[s[i-len(p)+1]]   # remove the count if it is 0
    return res

# Day 18 : Permutations in a String


def check_inclusion(s1, s2):
    import collections
    s1_char_map = collections.Counter(s1)
    s2_char_map = collections.Counter(s2[:len(s1)-1])
    for i in range(len(s1)-1, len(s2)):
        s2_char_map[s2[i]] += 1
        if s2_char_map == s1_char_map:
            return True
        s2_char_map[s2[i-len(s1)+1]] -= 1
        if s2_char_map[s2[i-len(s1)+1]] == 0:
            del s2_char_map[s2[i-len(s1)+1]]
    return False

# Day 19 : Online Stock Span


class StockSpanner:
    def __init__(self):
        self.history = []

    def next(self, price):
        weight = 1
        while self.history and self.history[-1][0] <= price:
            weight += self.history.pop()[1]
        self.history.append((price, weight))
        return weight
# Day 20 :  Kth Smallest Element in a BST


def kthSmallest(self, root: TreeNode, k: int) -> int:
    stack = []

    while True:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        k -= 1
        if not k:
            return root.val
        root = root.right

# Day 21  : Count Square Submatrices with All Ones


def countSquares(matrix):
    for i in range(1, len(matrix)):
        for j in range(1, len(matrix[0])):
            matrix[i][j] *= min(matrix[i - 1][j], matrix[i]
                                [j - 1], matrix[i - 1][j - 1]) + 1
    return sum(map(sum, matrix))


# Day 22: Sort Characters by Frequency

def frequency_sort(s):
    import collections
    return "".join([char * times for char, times in collections.Counter(str).most_common()])

# Day 23: Interval List Intersections


def intervalIntersection(A, B):
    ans = []
    i = j = 0

    while i < len(A) and j < len(B):
        # Let's check if A[i] intersects B[j].
        # lo - the startpoint of the intersection
        # hi - the endpoint of the intersection
        lo = max(A[i][0], B[j][0])
        hi = min(A[i][1], B[j][1])
        if lo <= hi:
            ans.append([lo, hi])

            # Remove the interval with the smallest endpoint
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1

    return ans

# Day 24: Construct BST from Preorder Traversal


def bstFromPreorder(preorder):
    from LeetCode_Archives.LC_data_struc import TreeNode
    import bisect

    def helper(i, j):
        if i == j:
            return None
        root = TreeNode(preorder[i])
        mid = bisect.bisect(preorder, preorder[i], i+1, j)
        root.left = helper(i+1, mid)
        root.right = helper(mid, j)
        return root
    return helper(0, len(preorder))

# Day 25: Uncrossed Lines


def maxUncrossedLines(A, B):
    m, n = len(A), len(B)
    dp = [0] * (n + 1)
    for i in range(m):
        for j in range(n)[::-1]:
            if A[i] == B[j]:
                dp[j + 1] = dp[j] + 1
        for j in range(n):
            dp[j + 1] = max(dp[j + 1], dp[j])
    return dp[n]

# Day 26: Contiguous Array


def findMaxLength(nums):
    #  add entry for initial count = 0 and index -1
    table = {0: -1}
    #  initialize max_length and count variable
    maxlen = count = 0

    for index, num in enumerate(nums):
        count += num or -1
        if count in table:
            maxlen = max(maxlen, index-table.get(count))
        else:
            table[count] = index
    return maxlen

# Day 27: Possible Bipartition


def possibleBipartion(N, dislakes):
    import collections
    graph = collections.defaultdict(list)
    for u, v in dislikes:
        graph[u].append(v)
        graph[v].append(u)

    color = {}

    def dfs(node, c=0):
        if node in color:
            return color[node] == c
        color[node] = c
        return all(dfs(nei, c ^ 1) for nei in graph[node])

    return all(dfs(node)
               for node in range(1, N+1)
               if node not in color)

# Day 28: Counting Bits


def countBits(num):
    res = [0]
    while len(res) <= num:
        res += [i+1 for i in res]
    return res[:num+1]

# Day 29: Course Schedule


def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    graph = [[] for _ in range(numCourses)]
    visited = [0 for _ in range(numCourses)]
    # create graph
    for pair in prerequisites:
        x, y = pair
        graph[x].append(y)
    # visit each node
    for i in range(numCourses):
        if not self.dfs(graph, visited, i):
            return False
    return True


def dfs(self, graph, visited, i):
    # if ith node is marked as being visited, then a cycle is found
    if visited[i] == -1:
        return False
    # if it is done visted, then do not visit again
    if visited[i] == 1:
        return True
    # mark as being visited
    visited[i] = -1
    # visit all the neighbours
    for j in graph[i]:
        if not self.dfs(graph, visited, j):
            return False
    # after visit all the neighbours, mark it as done visited
    visited[i] = 1
    return True


# Day 30:  K Closest Points to Origin
def kClosest(points, K):
    points.sort(key=lambda P: P[0]**2 + P[1]**2)
    return points[:K]
