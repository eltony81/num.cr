macro my_test(name, *args, prefix = "")
  puts {{name.stringify}}
  puts {{prefix.stringify}}
  {% for arg in args %}
    puts {{arg.stringify}}
  {% end %}
end

my_test("foo", 1, 2, 3, prefix: "bar")
