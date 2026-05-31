macro test(a, *args)
  # Correct way to use a macro variable in code generation
  puts "Call: {{a}}, {% for arg, index in args %}{{arg}}{% if index < args.size - 1 %}, {% end %}{% end %}"
end

test(1)
test(1, 2)
