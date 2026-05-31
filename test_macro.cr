macro test(*args)
  puts {{args.splat}}
end

test 1, 2, 3
