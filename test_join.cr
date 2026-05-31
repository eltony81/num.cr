macro test(a, *args)
  puts "Call: {{a}}{% if args.size > 0 %}, {{args.join(", ").id}}{% end %}"
end

test(1)
test(1, 2, 3)
