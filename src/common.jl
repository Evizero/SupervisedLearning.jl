export shuffleCols!

function safeRound(num)
  if VERSION < v"0.4-"
    iround(num)
  else
    round(Integer, num)
  end
end

function safeFloor(num)
  if VERSION < v"0.4-"
    ifloor(num)
  else
    floor(Integer, num)
  end
end

function shuffleCols!(A::Matrix)
  rows = size(A, 1)
  cols = size(A, 2)
  for c = 1:cols
    i = rand(c:cols)
    for r = 1:rows
      A[r,c], A[r,i] = A[r,i], A[r,c]
    end
  end
  A
end
