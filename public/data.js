export const challenges = [
  { level: 1, title: 'Reverse the shelf order', task: 'Invert the order of elements in an array.', example: '[3, 1, 9] → [9, 1, 3]' },
  { level: 2, title: 'Copy & sort prices', task: 'Copy an array into a new array and sort ascending. Do not modify original.', example: '[4, 2, 2, 9] → [2, 2, 4, 9]' },
  { level: 3, title: 'Find cheapest & priciest listing', task: 'Return (min, max) from an integer array.', example: '[8, 3, 11] → (3, 11)' },
  { level: 4, title: 'Cart total', task: 'Sum all numbers in an array.', example: '[5, 5, 10] → 20' },
  { level: 5, title: 'Average rating (rounded down)', task: 'Compute integer average (floor) of an array. Empty array → 0.', example: '[4, 5, 5] → 4' },
  { level: 6, title: 'Count a character in a title', task: 'Count occurrences of a character c in a string s (case-sensitive).', example: '("eBay best deal", "e") → 2' },
  { level: 7, title: 'Remove duplicate SKUs', task: 'Remove duplicates from an array while preserving order.', example: '[A, B, A, C, B] → [A, B, C]' },
  { level: 8, title: 'Merge two carts', task: 'Merge two integer arrays into one and sort ascending.', example: '[3,1] + [2,2] → [1,2,2,3]' },
  { level: 9, title: 'Rotate featured items', task: 'Rotate array right by k.', example: '[1,2,3,4,5], k=2 → [4,5,1,2,3]' },
  { level: 10, title: 'Title is a palindrome?', task: 'Return true if string reads same backward ignoring spaces and case.', example: '"Taco cat" → true' },
  { level: 11, title: 'Reverse words in a sentence', task: 'Reverse word order (trim extra spaces).', example: '"  ship   it  fast " → "fast it ship"' },
  { level: 12, title: 'Title Case', task: 'Capitalize first letter of each word, lower-case the rest.', example: '"hELLo woRLD" → "Hello World"' },
  { level: 13, title: 'First non-repeating character', task: 'Return first char that appears once; else return empty string.', example: '"swiss" → "w"' },
  { level: 14, title: 'Are these two listing titles anagrams?', task: 'Ignore spaces and case.', example: '"Debit Card" & "Bad Credit" → true' },
  { level: 15, title: 'eBayBuzz (FizzBuzz variant)', task: 'For 1..n: multiple of 3 → "e", multiple of 5 → "Bay", both → "eBay", else number as string.', example: '' },
  { level: 16, title: 'Apply a discount map', task: 'Given prices map id->price and discountPercent integer, return new map with discounted prices rounded down.', example: '{A:100}, 15 → {A:85}' },
  { level: 17, title: 'Extract order IDs from text', task: 'Given text containing tokens like ORD-12345, return all order IDs found.', example: '"Packed ORD-12 and ORD-999" → ["ORD-12","ORD-999"]' },
  { level: 18, title: 'Validate username', task: 'Valid if: 3–12 chars, only letters/digits/underscore, starts with letter.', example: '' },
  { level: 19, title: 'Parse a CSV row (simple)', task: 'Given "id,price,qty" return object/map with typed values.', example: '"SKU1,19,3" → {id:"SKU1", price:19, qty:3}' },
  { level: 20, title: 'Flatten a 2D inventory bin', task: 'Flatten 2D array into 1D.', example: '[[1,2],[3]] → [1,2,3]' },
  { level: 21, title: 'Transpose a matrix', task: 'Transpose rectangular matrix.', example: '[[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]' },
  { level: 22, title: 'Missing package number', task: 'Array contains numbers from 1..n with one missing; return missing.', example: '[1,2,4,5] (n=5) → 3' },
  { level: 23, title: 'Two-sum (gift card match)', task: 'Return indices of two numbers summing to target, or [-1,-1].', example: '[2,7,11,15], 9 → [0,1]' },
  { level: 24, title: 'Set intersection (shared watchers)', task: 'Intersection of two arrays, unique, sorted.', example: '[1,2,2,3] & [2,3,4] → [2,3]' },
  { level: 25, title: 'Group listings by first letter', task: 'Map first letter → list of titles (preserve input order).', example: '' },
  { level: 26, title: 'Remove vowels from a title', task: 'Remove a,e,i,o,u (both cases).', example: '"Auction" → "ctn"' },
  { level: 27, title: 'Fibonacci shipping code', task: 'Return nth Fibonacci (0-indexed: 0,1,1,2...).', example: '' },
  { level: 28, title: 'Prime badge', task: 'Return true if n is prime.', example: '' },
  { level: 29, title: 'All primes up to n', task: 'Return list of primes ≤ n (Sieve).', example: '' },
  { level: 30, title: 'Longest common prefix', task: 'Given array of strings, return longest common prefix.', example: '["ship","shine","shirt"] → "sh"' },
  { level: 31, title: 'Longest word length', task: 'Return length of longest word in a sentence (words split by spaces).', example: '' },
  { level: 32, title: 'Balanced parentheses', task: 'Validate a string containing ()[]{}.', example: '' },
  { level: 33, title: 'Postfix expression evaluator', task: 'Evaluate tokens like ["2","1","+","3","*"] → 9', example: '' },
  { level: 34, title: 'Queue using two stacks', task: 'Implement enqueue(x) and dequeue() using two stacks.', example: '' },
  { level: 35, title: 'Merge intervals (shipping windows)', task: 'Merge overlapping intervals.', example: '[[1,3],[2,6],[8,10]] → [[1,6],[8,10]]' },
  { level: 36, title: 'Binary search (price lookup)', task: 'Return index of target in sorted array else -1.', example: '' },
  { level: 37, title: 'First “peak” price', task: 'Return index i where arr[i] > arr[i-1] and arr[i] > arr[i+1] (if exists), else -1.', example: '' },
  { level: 38, title: 'Top K frequent words', task: 'Return top K words by frequency, tie-break alphabetical.', example: '' },
  { level: 39, title: 'Shortest path in warehouse grid (BFS)', task: 'Grid of 0/1; 0 = free, 1 = blocked. Find shortest steps from (0,0) to (r-1,c-1).', example: '' },
  { level: 40, title: 'Sort listings by multiple keys', task: 'Sort items by price asc, then rating desc, then id asc.', example: '' },
  { level: 41, title: 'Mini log analytics', task: 'Input lines "userId action"; return map action → count.', example: '' },
  { level: 42, title: 'Run-length encoding', task: '"aaabbc" → "a3b2c1"', example: '' },
  { level: 43, title: 'Decode run-length encoding', task: '"a3b2c1" → "aaabbc"', example: '' },
  { level: 44, title: 'Detect a cycle (linked list by index)', task: 'Given next[] where next[i] is next index or -1, return true if cycle exists.', example: '' },
  { level: 45, title: 'Edit distance (tiny DP)', task: 'Minimum edits (insert/delete/replace) to convert a→b.', example: '' },
  { level: 46, title: 'Coin change (minimum coins)', task: 'Return minimum coins to reach amount or -1 if impossible.', example: '' },
  { level: 47, title: 'Generate all permutations (small)', task: 'Return all permutations of a string (assume length ≤ 6).', example: '' },
  { level: 48, title: 'Subset sum (small)', task: 'Return true if any subset sums to target.', example: '' },
  { level: 49, title: '“Inventory diff”', task: 'Given expectedCounts and actualCounts maps (id->count), return list of ids where counts differ (missing treated as 0), sorted.', example: '' },
  { level: 50, title: 'Auction winner engine', task: 'Given bids as {itemId, bidderId, amount, time} pick winner per item. Highest amount wins; tie → earliest time wins.', example: '' }
];

export const referenceSolutions = {
  python: {
    1: `def lvl01_reverse_array(arr):\n    return list(reversed(arr))`,
    2: `def lvl02_copy_sort(arr):\n    return sorted(arr)`,
    3: `def lvl03_min_max(arr):\n    if not arr: return (None, None)\n    return (min(arr), max(arr))`,
    4: `def lvl04_sum(arr):\n    return sum(arr)`,
    5: `def lvl05_avg_floor(arr):\n    return 0 if not arr else (sum(arr) // len(arr))`,
    6: `def lvl06_count_char(s, c):\n    return sum(1 for ch in s if ch == c)`,
    7: `def lvl07_dedupe_keep_order(arr):\n    seen = set()\n    out = []\n    for x in arr:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out`,
    8: `def lvl08_merge_sort(a, b):\n    return sorted(a + b)`,
    9: `def lvl09_rotate_right(arr, k):\n    if not arr: return []\n    k %= len(arr)\n    return arr[-k:] + arr[:-k]`,
    10: `def lvl10_is_palindrome(s):\n    t = "".join(ch.lower() for ch in s if ch != " ")\n    return t == t[::-1]`,
    11: `def lvl11_reverse_words(s):\n    words = s.split()\n    return " ".join(reversed(words))`,
    12: `def lvl12_title_case(s):\n    return " ".join(w[:1].upper() + w[1:].lower() for w in s.split())`,
    13: `from collections import Counter\n\ndef lvl13_first_unique_char(s):\n    freq = Counter(s)\n    for ch in s:\n        if freq[ch] == 1:\n            return ch\n    return ""`,
    14: `def lvl14_are_anagrams(a, b):\n    sa = sorted(ch for ch in a.lower() if ch != " ")\n    sb = sorted(ch for ch in b.lower() if ch != " ")\n    return sa == sb`,
    15: `def lvl15_ebaybuzz(n):\n    out = []\n    for i in range(1, n+1):\n        s = ""\n        if i % 3 == 0: s += "e"\n        if i % 5 == 0: s += "Bay"\n        out.append(s if s else str(i))\n    return out`,
    16: `def lvl16_discount_map(prices, discount_percent):\n    out = {}\n    for k, v in prices.items():\n        out[k] = (v * (100 - discount_percent)) // 100\n    return out`,
    17: `import re\ndef lvl17_extract_order_ids(text):\n    return re.findall(r"\\bORD-\\d+\\b", text)`,
    18: `import re\ndef lvl18_valid_username(u):\n    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{2,11}", u))`,
    19: `def lvl19_parse_csv_row(line):\n    parts = [p.strip() for p in line.split(",")]\n    return {"id": parts[0], "price": int(parts[1]), "qty": int(parts[2])}`,
    20: `def lvl20_flatten(grid):\n    return [x for row in grid for x in row]`,
    21: `def lvl21_transpose(m):\n    if not m: return []\n    return [list(row) for row in zip(*m)]`,
    22: `def lvl22_missing(nums, n):\n    expected = n*(n+1)//2\n    return expected - sum(nums)`,
    23: `def lvl23_two_sum(nums, target):\n    seen = {}\n    for i, v in enumerate(nums):\n        need = target - v\n        if need in seen:\n            return [seen[need], i]\n        seen[v] = i\n    return [-1, -1]`,
    24: `def lvl24_intersection(a, b):\n    return sorted(set(a).intersection(b))`,
    25: `def lvl25_group_by_first_letter(titles):\n    out = {}\n    for t in titles:\n        if not t: continue\n        k = t[0]\n        out.setdefault(k, []).append(t)\n    return out`,
    26: `def lvl26_remove_vowels(s):\n    vowels = set("aeiouAEIOU")\n    return "".join(ch for ch in s if ch not in vowels)`,
    27: `def lvl27_fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return a`,
    28: `def lvl28_is_prime(n):\n    if n <= 1: return False\n    if n <= 3: return True\n    if n % 2 == 0 or n % 3 == 0: return False\n    i = 5\n    while i*i <= n:\n        if n % i == 0 or n % (i+2) == 0: return False\n        i += 6\n    return True`,
    29: `def lvl29_primes_upto(n):\n    if n < 2: return []\n    is_prime = [True]*(n+1)\n    is_prime[0]=is_prime[1]=False\n    p=2\n    while p*p<=n:\n        if is_prime[p]:\n            for k in range(p*p, n+1, p):\n                is_prime[k]=False\n        p+=1\n    return [i for i,v in enumerate(is_prime) if v]`,
    30: `def lvl30_lcp(words):\n    if not words: return ""\n    prefix = words[0]\n    for w in words[1:]:\n        while not w.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix: return ""\n    return prefix`,
    31: `def lvl31_longest_word_len(s):\n    return max((len(w) for w in s.split()), default=0)`,
    32: `def lvl32_balanced(s):\n    pairs = {')':'(', ']':'[', '}':'{'}\n    st = []\n    for ch in s:\n        if ch in "([{": st.append(ch)\n        elif ch in pairs:\n            if not st or st.pop() != pairs[ch]: return False\n    return not st`,
    33: `def lvl33_eval_postfix(tokens):\n    st=[]\n    for t in tokens:\n        if t in {"+","-","*","/"}:\n            b=st.pop(); a=st.pop()\n            if t=="+": st.append(a+b)\n            elif t=="-": st.append(a-b)\n            elif t=="*": st.append(a*b)\n            else: st.append(int(a/b))\n        else:\n            st.append(int(t))\n    return st[-1]`,
    34: `class Lvl34Queue:\n    def __init__(self):\n        self.a=[]; self.b=[]\n    def enqueue(self, x):\n        self.a.append(x)\n    def dequeue(self):\n        if not self.b:\n            while self.a: self.b.append(self.a.pop())\n        return None if not self.b else self.b.pop()`,
    35: `def lvl35_merge_intervals(intervals):\n    if not intervals: return []\n    intervals = sorted(intervals, key=lambda x: x[0])\n    out=[intervals[0][:]]\n    for s,e in intervals[1:]:\n        if s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s,e])\n    return out`,
    36: `def lvl36_binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == target: return mid\n        if arr[mid] < target: lo = mid+1\n        else: hi = mid-1\n    return -1`,
    37: `def lvl37_peak_index(arr):\n    for i in range(1, len(arr)-1):\n        if arr[i] > arr[i-1] and arr[i] > arr[i+1]:\n            return i\n    return -1`,
    38: `from collections import Counter\ndef lvl38_top_k_words(words, k):\n    freq = Counter(words)\n    return [w for w,_ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:k]]`,
    39: `from collections import deque\ndef lvl39_shortest_path(grid):\n    if not grid or not grid[0] or grid[0][0]==1: return -1\n    r, c = len(grid), len(grid[0])\n    if grid[r-1][c-1]==1: return -1\n    q = deque([(0,0,0)])\n    seen = {(0,0)}\n    while q:\n        x,y,d = q.popleft()\n        if (x,y) == (r-1,c-1): return d\n        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):\n            nx,ny = x+dx, y+dy\n            if 0<=nx<r and 0<=ny<c and grid[nx][ny]==0 and (nx,ny) not in seen:\n                seen.add((nx,ny))\n                q.append((nx,ny,d+1))\n    return -1`,
    40: `def lvl40_sort_items(items):\n    return sorted(items, key=lambda it: (it["price"], -it["rating"], it["id"]))`,
    41: `from collections import Counter\ndef lvl41_action_counts(lines):\n    actions = [ln.split()[1] for ln in lines if ln.strip()]\n    return dict(Counter(actions))`,
    42: `def lvl42_rle_encode(s):\n    if not s: return ""\n    out=[]\n    cur=s[0]; cnt=1\n    for ch in s[1:]:\n        if ch==cur: cnt+=1\n        else:\n            out.append(f"{cur}{cnt}")\n            cur=ch; cnt=1\n    out.append(f"{cur}{cnt}")\n    return "".join(out)`,
    43: `import re\ndef lvl43_rle_decode(s):\n    out=[]\n    for ch, num in re.findall(r"([A-Za-z])(\\d+)", s):\n        out.append(ch * int(num))\n    return "".join(out)`,
    44: `def lvl44_has_cycle(next_idx):\n    slow = 0\n    fast = 0\n    while fast != -1 and next_idx[fast] != -1:\n        slow = next_idx[slow]\n        fast = next_idx[next_idx[fast]]\n        if slow == -1 or fast == -1: return False\n        if slow == fast: return True\n    return False`,
    45: `def lvl45_edit_distance(a, b):\n    m, n = len(a), len(b)\n    dp = list(range(n+1))\n    for i in range(1, m+1):\n        prev = dp[0]\n        dp[0] = i\n        for j in range(1, n+1):\n            cur = dp[j]\n            if a[i-1] == b[j-1]:\n                dp[j] = prev\n            else:\n                dp[j] = 1 + min(prev, dp[j], dp[j-1])\n            prev = cur\n    return dp[n]`,
    46: `def lvl46_min_coins(coins, amount):\n    INF = 10**9\n    dp = [0] + [INF]*amount\n    for a in range(1, amount+1):\n        for c in coins:\n            if c <= a:\n                dp[a] = min(dp[a], dp[a-c] + 1)\n    return -1 if dp[amount] >= INF else dp[amount]`,
    47: `def lvl47_permutations(s):\n    out=[]\n    used=[False]*len(s)\n    path=[]\n    def backtrack():\n        if len(path)==len(s):\n            out.append("".join(path)); return\n        for i,ch in enumerate(s):\n            if used[i]: continue\n            used[i]=True; path.append(ch)\n            backtrack()\n            path.pop(); used[i]=False\n    backtrack()\n    return out`,
    48: `def lvl48_subset_sum(nums, target):\n    possible = {0}\n    for x in nums:\n        possible |= {p + x for p in possible}\n    return target in possible`,
    49: `def lvl49_inventory_diff(expected, actual):\n    keys = set(expected) | set(actual)\n    bad = [k for k in keys if expected.get(k,0) != actual.get(k,0)]\n    return sorted(bad)`,
    50: `def lvl50_auction_winners(bids):\n    best = {}\n    for b in bids:\n        item = b["itemId"]\n        if item not in best:\n            best[item] = b\n        else:\n            cur = best[item]\n            if (b["amount"] > cur["amount"]) or (b["amount"] == cur["amount"] and b["time"] < cur["time"]):\n                best[item] = b\n    return {item: b["bidderId"] for item, b in best.items()}`
  },
  javascript: {
    1: `function lvl01ReverseArray(arr) {\n  return arr.slice().reverse();\n}`,
    2: `function lvl02CopySort(arr) {\n  return arr.slice().sort((a,b) => a-b);\n}`,
    3: `function lvl03MinMax(arr) {\n  if (arr.length === 0) return [null, null];\n  return [Math.min(...arr), Math.max(...arr)];\n}`,
    4: `function lvl04Sum(arr) {\n  return arr.reduce((a,v) => a+v, 0);\n}`,
    5: `function lvl05AvgFloor(arr) {\n  if (arr.length === 0) return 0;\n  const s = arr.reduce((a,v)=>a+v,0);\n  return Math.floor(s / arr.length);\n}`,
    6: `function lvl06CountChar(s, c) {\n  let cnt = 0;\n  for (const ch of s) if (ch === c) cnt++;\n  return cnt;\n}`,
    7: `function lvl07DedupeKeepOrder(arr) {\n  const seen = new Set();\n  const out = [];\n  for (const x of arr) if (!seen.has(x)) { seen.add(x); out.push(x); }\n  return out;\n}`,
    8: `function lvl08MergeSort(a, b) {\n  return a.concat(b).slice().sort((x,y)=>x-y);\n}`,
    9: `function lvl09RotateRight(arr, k) {\n  const n = arr.length;\n  if (n === 0) return [];\n  k = ((k % n) + n) % n;\n  return arr.slice(n - k).concat(arr.slice(0, n - k));\n}`,
    10: `function lvl10IsPalindrome(s) {\n  const t = s.replaceAll(" ", "").toLowerCase();\n  return t === t.split("").reverse().join("");\n}`,
    11: `function lvl11ReverseWords(s) {\n  return s.trim().split(/\s+/).reverse().join(" ");\n}`,
    12: `function lvl12TitleCase(s) {\n  return s.trim().split(/\s+/).map(w =>\n    w.length ? w[0].toUpperCase() + w.slice(1).toLowerCase() : w\n  ).join(" ");\n}`,
    13: `function lvl13FirstUniqueChar(s) {\n  const freq = new Map();\n  for (const ch of s) freq.set(ch, (freq.get(ch)||0)+1);\n  for (const ch of s) if (freq.get(ch) === 1) return ch;\n  return "";\n}`,
    14: `function lvl14AreAnagrams(a, b) {\n  const norm = s => s.replaceAll(" ", "").toLowerCase().split("").sort().join("");\n  return norm(a) === norm(b);\n}`,
    15: `function lvl15EbayBuzz(n) {\n  const out = [];\n  for (let i=1;i<=n;i++){\n    let s = "";\n    if (i%3===0) s += "e";\n    if (i%5===0) s += "Bay";\n    out.push(s || String(i));\n  }\n  return out;\n}`,
    16: `function lvl16DiscountMap(prices, discountPercent) {\n  const out = {};\n  for (const [k,v] of Object.entries(prices)) {\n    out[k] = Math.floor(v * (100 - discountPercent) / 100);\n  }\n  return out;\n}`,
    17: `function lvl17ExtractOrderIds(text) {\n  return text.match(/\bORD-\d+\b/g) ?? [];\n}`,
    18: `function lvl18ValidUsername(u) {\n  return /^[A-Za-z][A-Za-z0-9_]{2,11}$/.test(u);\n}`,
    19: `function lvl19ParseCsvRow(line) {\n  const [id, price, qty] = line.split(",").map(s => s.trim());\n  return { id, price: Number(price), qty: Number(qty) };\n}`,
    20: `function lvl20Flatten(grid) {\n  return grid.flat();\n}`,
    21: `function lvl21Transpose(m) {\n  if (m.length === 0) return [];\n  return m[0].map((_, j) => m.map(row => row[j]));\n}`,
    22: `function lvl22Missing(nums, n) {\n  const expected = n*(n+1)/2;\n  const actual = nums.reduce((a,v)=>a+v,0);\n  return expected - actual;\n}`,
    23: `function lvl23TwoSum(nums, target) {\n  const seen = new Map();\n  for (let i=0;i<nums.length;i++){\n    const need = target - nums[i];\n    if (seen.has(need)) return [seen.get(need), i];\n    seen.set(nums[i], i);\n  }\n  return [-1,-1];\n}`,
    24: `function lvl24Intersection(a, b) {\n  const sa = new Set(a);\n  const inter = [...new Set(b.filter(x => sa.has(x)))];\n  return inter.sort((x,y)=>x-y);\n}`,
    25: `function lvl25GroupByFirstLetter(titles) {\n  const out = {};\n  for (const t of titles) {\n    if (!t) continue;\n    const k = t[0];\n    (out[k] ??= []).push(t);\n  }\n  return out;\n}`,
    26: `function lvl26RemoveVowels(s) {\n  return s.replace(/[aeiou]/gi, "");\n}`,
    27: `function lvl27Fib(n) {\n  let a=0, b=1;\n  for (let i=0;i<n;i++){ [a,b] = [b, a+b]; }\n  return a;\n}`,
    28: `function lvl28IsPrime(n) {\n  if (n<=1) return false;\n  if (n<=3) return true;\n  if (n%2===0 || n%3===0) return false;\n  for (let i=5; i*i<=n; i+=6)\n    if (n%i===0 || n%(i+2)===0) return false;\n  return true;\n}`,
    29: `function lvl29PrimesUpto(n) {\n  if (n < 2) return [];\n  const isPrime = Array(n+1).fill(true);\n  isPrime[0]=false; isPrime[1]=false;\n  for (let p=2; p*p<=n; p++){\n    if (!isPrime[p]) continue;\n    for (let k=p*p; k<=n; k+=p) isPrime[k]=false;\n  }\n  const out=[];\n  for (let i=2;i<=n;i++) if (isPrime[i]) out.push(i);\n  return out;\n}`,
    30: `function lvl30Lcp(words) {\n  if (words.length === 0) return "";\n  let prefix = words[0];\n  for (let i=1;i<words.length;i++){\n    while (!words[i].startsWith(prefix)) {\n      prefix = prefix.slice(0,-1);\n      if (!prefix) return "";\n    }\n  }\n  return prefix;\n}`,
    31: `function lvl31LongestWordLen(s) {\n  const parts = s.trim() ? s.trim().split(/\s+/) : [];\n  return parts.reduce((m,w)=>Math.max(m,w.length), 0);\n}`,
    32: `function lvl32Balanced(s) {\n  const pairs = new Map([ [')','('], [']','['], ['}','{'] ]);\n  const st = [];\n  for (const ch of s) {\n    if ("([{".includes(ch)) st.push(ch);\n    else if (pairs.has(ch)) {\n      if (!st.length || st.pop() !== pairs.get(ch)) return false;\n    }\n  }\n  return st.length === 0;\n}`,
    33: `function lvl33EvalPostfix(tokens) {\n  const st = [];\n  for (const t of tokens) {\n    if (["+","-","*","/"].includes(t)) {\n      const b = st.pop(), a = st.pop();\n      if (t === "+") st.push(a+b);\n      else if (t === "-") st.push(a-b);\n      else if (t === "*") st.push(a*b);\n      else st.push((a/b) | 0);\n    } else st.push(Number(t));\n  }\n  return st[st.length-1];\n}`,
    34: `class Lvl34Queue {\n  constructor(){ this.a=[]; this.b=[]; }\n  enqueue(x){ this.a.push(x); }\n  dequeue(){\n    if (!this.b.length) while (this.a.length) this.b.push(this.a.pop());\n    return this.b.length ? this.b.pop() : null;\n  }\n}`,
    35: `function lvl35MergeIntervals(intervals) {\n  if (!intervals.length) return [];\n  intervals = intervals.slice().sort((a,b)=>a[0]-b[0]);\n  const out = [intervals[0].slice()];\n  for (const [s,e] of intervals.slice(1)) {\n    const last = out[out.length-1];\n    if (s <= last[1]) last[1] = Math.max(last[1], e);\n    else out.push([s,e]);\n  }\n  return out;\n}`,
    36: `function lvl36BinarySearch(arr, target) {\n  let lo=0, hi=arr.length-1;\n  while (lo<=hi){\n    const mid = (lo+hi)>>1;\n    if (arr[mid]===target) return mid;\n    if (arr[mid]<target) lo=mid+1;\n    else hi=mid-1;\n  }\n  return -1;\n}`,
    37: `function lvl37PeakIndex(arr) {\n  for (let i=1;i<arr.length-1;i++)\n    if (arr[i]>arr[i-1] && arr[i]>arr[i+1]) return i;\n  return -1;\n}`,
    38: `function lvl38TopKWords(words, k) {\n  const freq = new Map();\n  for (const w of words) freq.set(w, (freq.get(w)||0)+1);\n  return [...freq.entries()]\n    .sort((a,b)=> (b[1]-a[1]) || a[0].localeCompare(b[0]))\n    .slice(0,k)\n    .map(([w])=>w);\n}`,
    39: `function lvl39ShortestPath(grid) {\n  const r = grid.length;\n  if (!r) return -1;\n  const c = grid[0].length;\n  if (grid[0][0]===1 || grid[r-1][c-1]===1) return -1;\n  const dist = Array.from({length:r}, ()=>Array(c).fill(-1));\n  const q = [[0,0]];\n  dist[0][0]=0;\n  const dirs = [[1,0],[-1,0],[0,1],[0,-1]];\n  let qi=0;\n  while (qi<q.length){\n    const [x,y]=q[qi++];\n    if (x===r-1 && y===c-1) return dist[x][y];\n    for (const [dx,dy] of dirs){\n      const nx=x+dx, ny=y+dy;\n      if (0<=nx && nx<r && 0<=ny && ny<c && grid[nx][ny]===0 && dist[nx][ny]===-1){\n        dist[nx][ny]=dist[x][y]+1;\n        q.push([nx,ny]);\n      }\n    }\n  }\n  return -1;\n}`,
    40: `function lvl40SortItems(items) {\n  return items.slice().sort((a,b)=>\n    (a.price-b.price) || (b.rating-a.rating) || a.id.localeCompare(b.id)\n  );\n}`,
    41: `function lvl41ActionCounts(lines) {\n  const out = {};\n  for (const ln of lines) {\n    const parts = (ln ?? "").trim().split(/\s+/);\n    if (parts.length < 2) continue;\n    const action = parts[1];\n    out[action] = (out[action] ?? 0) + 1;\n  }\n  return out;\n}`,
    42: `function lvl42RleEncode(s) {\n  if (!s) return "";\n  let out = "";\n  let cur = s[0], cnt = 1;\n  for (let i=1;i<s.length;i++){\n    const ch = s[i];\n    if (ch===cur) cnt++;\n    else { out += cur + String(cnt); cur = ch; cnt = 1; }\n  }\n  out += cur + String(cnt);\n  return out;\n}`,
    43: `function lvl43RleDecode(s) {\n  let out = "";\n  for (const m of s.matchAll(/([A-Za-z])(\d+)/g)) {\n    out += m[1].repeat(Number(m[2]));\n  }\n  return out;\n}`,
    44: `function lvl44HasCycle(next) {\n  let slow = 0, fast = 0;\n  while (fast !== -1 && next[fast] !== -1) {\n    slow = next[slow];\n    fast = next[next[fast]];\n    if (slow === -1 || fast === -1) return false;\n    if (slow === fast) return true;\n  }\n  return false;\n}`,
    45: `function lvl45EditDistance(a, b) {\n  const m=a.length, n=b.length;\n  const dp = Array.from({length:n+1}, (_,j)=>j);\n  for (let i=1;i<=m;i++){
    let prev = dp[0];
    dp[0]=i;
    for (let j=1;j<=n;j++){
      const cur = dp[j];
      dp[j] = (a[i-1]===b[j-1]) ? prev : 1 + Math.min(prev, dp[j], dp[j-1]);
      prev = cur;
    }
  }
  return dp[n];
}`,
    46: `function lvl46MinCoins(coins, amount) {
  const INF = 1e9;
  const dp = Array(amount+1).fill(INF);
  dp[0]=0;
  for (let a=1;a<=amount;a++){
    for (const c of coins){
      if (c<=a) dp[a] = Math.min(dp[a], dp[a-c]+1);
    }
  }
  return dp[amount]>=INF ? -1 : dp[amount];
}`,
    47: `function lvl47Permutations(s) {
  const out = [];
  const used = Array(s.length).fill(false);
  const path = [];
  function bt(){
    if (path.length === s.length){ out.push(path.join("")); return; }
    for (let i=0;i<s.length;i++){
      if (used[i]) continue;
      used[i]=true; path.push(s[i]);
      bt();
      path.pop(); used[i]=false;
    }
  }
  bt();
  return out;
}`,
    48: `function lvl48SubsetSum(nums, target) {
  let possible = new Set([0]);
  for (const x of nums) {
    const next = new Set(possible);
    for (const p of possible) next.add(p + x);
    possible = next;
  }
  return possible.has(target);
}`,
    49: `function lvl49InventoryDiff(expected, actual) {
  const keys = new Set([...Object.keys(expected), ...Object.keys(actual)]);
  const bad = [];
  for (const k of keys) {
    const e = expected[k] ?? 0;
    const a = actual[k] ?? 0;
    if (e !== a) bad.push(k);
  }
  return bad.sort();
}`,
    50: `function lvl50AuctionWinners(bids) {
  const best = new Map();
  for (const b of bids) {
    const cur = best.get(b.itemId);
    if (!cur || b.amount > cur.amount || (b.amount === cur.amount && b.time < cur.time)) {
      best.set(b.itemId, b);
    }
  }
  const out = {};
  for (const [item, bid] of best.entries()) out[item] = bid.bidderId;
  return out;
}`
  },
  java: {
    1: `static int[] lvl01ReverseArray(int[] arr) {\n    int[] out = Arrays.copyOf(arr, arr.length);\n    for (int i = 0, j = out.length - 1; i < j; i++, j--) {\n        int tmp = out[i]; out[i] = out[j]; out[j] = tmp;\n    }\n    return out;\n}`,
    2: `static int[] lvl02CopySort(int[] arr) {\n    int[] out = Arrays.copyOf(arr, arr.length);\n    Arrays.sort(out);\n    return out;\n}`,
    3: `static int[] lvl03MinMax(int[] arr) {\n    if (arr.length == 0) return new int[]{Integer.MIN_VALUE, Integer.MAX_VALUE};\n    int mn = arr[0], mx = arr[0];\n    for (int v : arr) { mn = Math.min(mn, v); mx = Math.max(mx, v); }\n    return new int[]{mn, mx};\n}`,
    4: `static int lvl04Sum(int[] arr) {\n    int s = 0;\n    for (int v : arr) s += v;\n    return s;\n}`,
    5: `static int lvl05AvgFloor(int[] arr) {\n    if (arr.length == 0) return 0;\n    long s = 0;\n    for (int v : arr) s += v;\n    return (int)(s / arr.length);\n}`,
    6: `static int lvl06CountChar(String s, char c) {\n    int cnt = 0;\n    for (int i = 0; i < s.length(); i++) if (s.charAt(i) == c) cnt++;\n    return cnt;\n}`,
    7: `static <T> List<T> lvl07DedupeKeepOrder(List<T> arr) {\n    Set<T> seen = new HashSet<>();\n    List<T> out = new ArrayList<>();\n    for (T x : arr) if (seen.add(x)) out.add(x);\n    return out;\n}`,
    8: `static int[] lvl08MergeSort(int[] a, int[] b) {\n    int[] out = new int[a.length + b.length];\n    System.arraycopy(a, 0, out, 0, a.length);\n    System.arraycopy(b, 0, out, a.length, b.length);\n    Arrays.sort(out);\n    return out;\n}`,
    9: `static int[] lvl09RotateRight(int[] arr, int k) {\n    int n = arr.length;\n    if (n == 0) return new int[0];\n    k %= n;\n    int[] out = new int[n];\n    for (int i = 0; i < n; i++) out[(i + k) % n] = arr[i];\n    return out;\n}`,
    10: `static boolean lvl10IsPalindrome(String s) {\n    StringBuilder sb = new StringBuilder();\n    for (int i = 0; i < s.length(); i++) {\n        char ch = s.charAt(i);\n        if (ch != ' ') sb.append(Character.toLowerCase(ch));\n    }\n    String t = sb.toString();\n    return new StringBuilder(t).reverse().toString().equals(t);\n}`,
    11: `static String lvl11ReverseWords(String s) {\n    String[] parts = s.trim().split("\\s+");\n    List<String> list = Arrays.asList(parts);\n    Collections.reverse(list);\n    return String.join(" ", list);\n}`,
    12: `static String lvl12TitleCase(String s) {\n    String[] parts = s.trim().split("\\s+");\n    for (int i = 0; i < parts.length; i++) {\n        String w = parts[i];\n        parts[i] = w.isEmpty() ? w : (w.substring(0,1).toUpperCase() + w.substring(1).toLowerCase());\n    }\n    return String.join(" ", parts);\n}`,
    13: `static String lvl13FirstUniqueChar(String s) {\n    Map<Character, Integer> freq = new HashMap<>();\n    for (char c : s.toCharArray()) freq.put(c, freq.getOrDefault(c, 0) + 1);\n    for (char c : s.toCharArray()) if (freq.get(c) == 1) return String.valueOf(c);\n    return "";\n}`,
    14: `static boolean lvl14AreAnagrams(String a, String b) {\n    char[] ca = a.replace(" ", "").toLowerCase().toCharArray();\n    char[] cb = b.replace(" ", "").toLowerCase().toCharArray();\n    Arrays.sort(ca); Arrays.sort(cb);\n    return Arrays.equals(ca, cb);\n}`,
    15: `static List<String> lvl15EbayBuzz(int n) {\n    List<String> out = new ArrayList<>();\n    for (int i = 1; i <= n; i++) {\n        String s = "";\n        if (i % 3 == 0) s += "e";\n        if (i % 5 == 0) s += "Bay";\n        out.add(s.isEmpty() ? String.valueOf(i) : s);\n    }\n    return out;\n}`,
    16: `static Map<String,Integer> lvl16DiscountMap(Map<String,Integer> prices, int discountPercent) {\n    Map<String,Integer> out = new HashMap<>();\n    for (var e : prices.entrySet()) {\n        int v = e.getValue();\n        out.put(e.getKey(), (v * (100 - discountPercent)) / 100);\n    }\n    return out;\n}`,
    17: `static List<String> lvl17ExtractOrderIds(String text) {\n    List<String> out = new ArrayList<>();\n    var m = java.util.regex.Pattern.compile("\\bORD-\\d+\\b").matcher(text);\n    while (m.find()) out.add(m.group());\n    return out;\n}`,
    18: `static boolean lvl18ValidUsername(String u) {\n    return u.matches("[A-Za-z][A-Za-z0-9_]{2,11}");\n}`,
    19: `static Map<String,Object> lvl19ParseCsvRow(String line) {\n    String[] p = line.split(",");\n    Map<String,Object> out = new HashMap<>();\n    out.put("id", p[0].trim());\n    out.put("price", Integer.parseInt(p[1].trim()));\n    out.put("qty", Integer.parseInt(p[2].trim()));\n    return out;\n}`,
    20: `static List<Integer> lvl20Flatten(List<List<Integer>> grid) {\n    List<Integer> out = new ArrayList<>();\n    for (var row : grid) out.addAll(row);\n    return out;\n}`,
    21: `static int[][] lvl21Transpose(int[][] m) {\n    if (m.length == 0) return new int[0][0];\n    int r = m.length, c = m[0].length;\n    int[][] out = new int[c][r];\n    for (int i=0;i<r;i++) for (int j=0;j<c;j++) out[j][i] = m[i][j];\n    return out;\n}`,
    22: `static int lvl22Missing(int[] nums, int n) {\n    long expected = (long)n*(n+1)/2;\n    long actual = 0;\n    for (int v : nums) actual += v;\n    return (int)(expected - actual);\n}`,
    23: `static int[] lvl23TwoSum(int[] nums, int target) {\n    Map<Integer,Integer> seen = new HashMap<>();\n    for (int i=0;i<nums.length;i++) {\n        int need = target - nums[i];\n        if (seen.containsKey(need)) return new int[]{seen.get(need), i};\n        seen.put(nums[i], i);\n    }\n    return new int[]{-1,-1};\n}`,
    24: `static int[] lvl24Intersection(int[] a, int[] b) {\n    Set<Integer> sa = new HashSet<>();\n    for (int v : a) sa.add(v);\n    Set<Integer> inter = new HashSet<>();\n    for (int v : b) if (sa.contains(v)) inter.add(v);\n    int[] out = inter.stream().mapToInt(x->x).toArray();\n    Arrays.sort(out);\n    return out;\n}`,
    25: `static Map<Character,List<String>> lvl25GroupByFirstLetter(List<String> titles) {\n    Map<Character,List<String>> out = new HashMap<>();\n    for (String t : titles) {\n        if (t == null || t.isEmpty()) continue;\n        char k = t.charAt(0);\n        out.computeIfAbsent(k, __ -> new ArrayList<>()).add(t);\n    }\n    return out;\n}`,
    26: `static String lvl26RemoveVowels(String s) {\n    String vowels = "aeiouAEIOU";\n    StringBuilder out = new StringBuilder();\n    for (char c : s.toCharArray()) if (vowels.indexOf(c) < 0) out.append(c);\n    return out.toString();\n}`,
    27: `static long lvl27Fib(int n) {\n    long a=0, b=1;\n    for (int i=0;i<n;i++) { long t=a; a=b; b=t+b; }\n    return a;\n}`,
    28: `static boolean lvl28IsPrime(int n) {\n    if (n <= 1) return false;\n    if (n <= 3) return true;\n    if (n % 2 == 0 || n % 3 == 0) return false;\n    for (int i=5; i*(long)i<=n; i+=6)\n        if (n%i==0 || n%(i+2)==0) return false;\n    return true;\n}`,
    29: `static List<Integer> lvl29PrimesUpto(int n) {\n    if (n < 2) return List.of();\n    boolean[] isPrime = new boolean[n+1];\n    Arrays.fill(isPrime, true);\n    isPrime[0]=false; isPrime[1]=false;\n    for (int p=2; p*p<=n; p++) if (isPrime[p])\n        for (int k=p*p; k<=n; k+=p) isPrime[k]=false;\n    List<Integer> out = new ArrayList<>();\n    for (int i=2;i<=n;i++) if (isPrime[i]) out.add(i);\n    return out;\n}`,
    30: `static String lvl30Lcp(String[] words) {\n    if (words.length == 0) return "";\n    String prefix = words[0];\n    for (int i=1;i<words.length;i++) {\n        while (!words[i].startsWith(prefix)) {\n            prefix = prefix.substring(0, prefix.length()-1);\n            if (prefix.isEmpty()) return "";\n        }\n    }\n    return prefix;\n}`,
    31: `static int lvl31LongestWordLen(String s) {\n    String[] parts = s.trim().isEmpty() ? new String[0] : s.trim().split("\\s+");\n    int mx = 0;\n    for (String w : parts) mx = Math.max(mx, w.length());\n    return mx;\n}`,
    32: `static boolean lvl32Balanced(String s) {\n    Map<Character,Character> pairs = Map.of(')','(',']','[','}','{');\n    Deque<Character> st = new ArrayDeque<>();\n    for (char ch : s.toCharArray()) {\n        if (ch=='('||ch=='['||ch=='{') st.push(ch);\n        else if (pairs.containsKey(ch)) {\n            if (st.isEmpty() || st.pop() != pairs.get(ch)) return false;\n        }\n    }\n    return st.isEmpty();\n}`,
    33: `static int lvl33EvalPostfix(String[] tokens) {\n    Deque<Integer> st = new ArrayDeque<>();\n    for (String t : tokens) {\n        if ("+-*/".contains(t) && t.length()==1) {\n            int b = st.pop(), a = st.pop();\n            switch (t) {\n                case "+": st.push(a+b); break;\n                case "-": st.push(a-b); break;\n                case "*": st.push(a*b); break;\n                default: st.push(a/b); break;\n            }\n        } else st.push(Integer.parseInt(t));\n    }\n    return st.peek();\n}`,
    34: `static class Lvl34Queue {\n    Deque<Integer> a = new ArrayDeque<>();\n    Deque<Integer> b = new ArrayDeque<>();\n    void enqueue(int x){ a.push(x); }\n    Integer dequeue(){\n        if (b.isEmpty()) while (!a.isEmpty()) b.push(a.pop());\n        return b.isEmpty() ? null : b.pop();\n    }\n}`,
    35: `static int[][] lvl35MergeIntervals(int[][] intervals) {\n    if (intervals.length == 0) return new int[0][0];\n    Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));\n    List<int[]> out = new ArrayList<>();\n    out.add(new int[]{intervals[0][0], intervals[0][1]});\n    for (int i=1;i<intervals.length;i++){\n        int s = intervals[i][0], e = intervals[i][1];\n        int[] last = out.get(out.size()-1);\n        if (s <= last[1]) last[1] = Math.max(last[1], e);\n        else out.add(new int[]{s,e});\n    }\n    return out.toArray(new int[out.size()][]);\n}`,
    36: `static int lvl36BinarySearch(int[] arr, int target) {\n    int lo=0, hi=arr.length-1;\n    while (lo<=hi) {\n        int mid = lo + (hi-lo)/2;\n        if (arr[mid]==target) return mid;\n        if (arr[mid]<target) lo=mid+1;\n        else hi=mid-1;\n    }\n    return -1;\n}`,
    37: `static int lvl37PeakIndex(int[] arr) {\n    for (int i=1;i<arr.length-1;i++)\n        if (arr[i]>arr[i-1] && arr[i]>arr[i+1]) return i;\n    return -1;\n}`,
    38: `static List<String> lvl38TopKWords(List<String> words, int k) {\n    Map<String,Integer> freq = new HashMap<>();\n    for (String w: words) freq.put(w, freq.getOrDefault(w,0)+1);\n    List<String> keys = new ArrayList<>(freq.keySet());\n    keys.sort((a,b) -> {\n        int ca = freq.get(a), cb = freq.get(b);\n        if (ca != cb) return Integer.compare(cb, ca);\n        return a.compareTo(b);\n    });\n    return keys.subList(0, Math.min(k, keys.size()));\n}`,
    39: `static int lvl39ShortestPath(int[][] grid) {\n    int r = grid.length;\n    if (r==0) return -1;\n    int c = grid[0].length;\n    if (grid[0][0]==1 || grid[r-1][c-1]==1) return -1;\n    int[][] dist = new int[r][c];\n    for (int[] row : dist) Arrays.fill(row, -1);\n    ArrayDeque<int[]> q = new ArrayDeque<>();\n    q.add(new int[]{0,0});\n    dist[0][0]=0;\n    int[] dx={1,-1,0,0}, dy={0,0,1,-1};\n    while(!q.isEmpty()){
        int[] cur=q.poll();
        int x=cur[0], y=cur[1];
        if (x==r-1 && y==c-1) return dist[x][y];
        for (int i=0;i<4;i++){
            int nx=x+dx[i], ny=y+dy[i];
            if (0<=nx && nx<r && 0<=ny && ny<c && grid[nx][ny]==0 && dist[nx][ny]==-1){
                dist[nx][ny]=dist[x][y]+1;
                q.add(new int[]{nx,ny});
            }
        }
    }
    return -1;
}`,
    40: `static class Item {
    String id; int price; double rating;
    Item(String id,int price,double rating){ this.id=id; this.price=price; this.rating=rating; }
}
static List<Item> lvl40SortItems(List<Item> items) {
    List<Item> out = new ArrayList<>(items);
    out.sort((a,b)->{
        if (a.price != b.price) return Integer.compare(a.price, b.price);
        int r = Double.compare(b.rating, a.rating);
        if (r != 0) return r;
        return a.id.compareTo(b.id);
    });
    return out;
}`,
    41: `static Map<String,Integer> lvl41ActionCounts(List<String> lines) {
    Map<String,Integer> out = new HashMap<>();
    for (String ln : lines) {
        if (ln == null || ln.trim().isEmpty()) continue;
        String[] p = ln.trim().split("\\s+");
        if (p.length < 2) continue;
        String action = p[1];
        out.put(action, out.getOrDefault(action,0)+1);
    }
    return out;
}`,
    42: `static String lvl42RleEncode(String s) {
    if (s.isEmpty()) return "";
    StringBuilder out = new StringBuilder();
    char cur = s.charAt(0);
    int cnt = 1;
    for (int i=1;i<s.length();i++){
        char ch = s.charAt(i);
        if (ch==cur) cnt++;
        else {
            out.append(cur).append(cnt);
            cur = ch; cnt = 1;
        }
    }
    out.append(cur).append(cnt);
    return out.toString();
}`,
    43: `static String lvl43RleDecode(String s) {
    StringBuilder out = new StringBuilder();
    var m = java.util.regex.Pattern.compile("([A-Za-z])(\\d+)").matcher(s);
    while (m.find()) {
        char ch = m.group(1).charAt(0);
        int n = Integer.parseInt(m.group(2));
        out.append(String.valueOf(ch).repeat(n));
    }
    return out.toString();
}`,
    44: `static boolean lvl44HasCycle(int[] next) {
    int slow = 0, fast = 0;
    while (fast != -1 && next[fast] != -1) {
        slow = next[slow];
        fast = next[next[fast]];
        if (slow == -1 || fast == -1) return false;
        if (slow == fast) return true;
    }
    return false;
}`,
    45: `static int lvl45EditDistance(String a, String b) {
    int m=a.length(), n=b.length();
    int[] dp = new int[n+1];
    for (int j=0;j<=n;j++) dp[j]=j;
    for (int i=1;i<=m;i++){
        int prev = dp[0];
        dp[0]=i;
        for (int j=1;j<=n;j++){
            int cur = dp[j];
            if (a.charAt(i-1)==b.charAt(j-1)) dp[j]=prev;
            else dp[j]=1+Math.min(prev, Math.min(dp[j], dp[j-1]));
            prev = cur;
        }
    }
    return dp[n];
}`,
    46: `static int lvl46MinCoins(int[] coins, int amount) {
    int INF = 1_000_000_000;
    int[] dp = new int[amount+1];
    Arrays.fill(dp, INF);
    dp[0]=0;
    for (int a=1;a<=amount;a++){
        for (int c: coins){
            if (c<=a) dp[a] = Math.min(dp[a], dp[a-c]+1);
        }
    }
    return dp[amount]>=INF ? -1 : dp[amount];
}`,
    47: `static List<String> lvl47Permutations(String s) {
    List<String> out = new ArrayList<>();
    boolean[] used = new boolean[s.length()];
    StringBuilder path = new StringBuilder();
    class BT {
        void run() {
            if (path.length() == s.length()) { out.add(path.toString()); return; }
            for (int i=0;i<s.length();i++){
                if (used[i]) continue;
                used[i]=true; path.append(s.charAt(i));
                run();
                path.deleteCharAt(path.length()-1); used[i]=false;
            }
        }
    }
    new BT().run();
    return out;
}`,
    48: `static boolean lvl48SubsetSum(int[] nums, int target) {
    Set<Integer> possible = new HashSet<>();
    possible.add(0);
    for (int x : nums) {
        Set<Integer> next = new HashSet<>(possible);
        for (int p : possible) next.add(p + x);
        possible = next;
    }
    return possible.contains(target);
}`,
    49: `static List<String> lvl49InventoryDiff(Map<String,Integer> expected, Map<String,Integer> actual) {
    Set<String> keys = new HashSet<>();
    keys.addAll(expected.keySet());
    keys.addAll(actual.keySet());
    List<String> bad = new ArrayList<>();
    for (String k : keys) {
        int e = expected.getOrDefault(k,0);
        int a = actual.getOrDefault(k,0);
        if (e != a) bad.add(k);
    }
    Collections.sort(bad);
    return bad;
}`,
    50: `static class Bid {
    String itemId, bidderId; int amount; long time;
    Bid(String itemId,String bidderId,int amount,long time){
        this.itemId=itemId; this.bidderId=bidderId; this.amount=amount; this.time=time;
    }
}
static Map<String,String> lvl50AuctionWinners(List<Bid> bids) {
    Map<String,Bid> best = new HashMap<>();
    for (Bid b : bids) {
        Bid cur = best.get(b.itemId);
        if (cur == null ||
            b.amount > cur.amount ||
            (b.amount == cur.amount && b.time < cur.time)
        ) best.put(b.itemId, b);
    }
    Map<String,String> out = new HashMap<>();
    for (var e : best.entrySet()) out.put(e.getKey(), e.getValue().bidderId);
    return out;
}`
  }
};

export const quotes = [
  '“Success is the sum of small efforts, repeated day in and day out.” – Robert Collier',
  '“Courage is resistance to fear, mastery of fear—not absence of fear.” – Mark Twain',
  '“Great things are done by a series of small things brought together.” – Vincent van Gogh',
  '“If you want to go fast, go alone. If you want to go far, go together.” – African Proverb',
  '“The best error message is the one that never shows up.” – Thomas Fuchs'
];
