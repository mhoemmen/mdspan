mdspan_add_test(mdspan.assign.pass)
mdspan_add_test(mdspan.conversion.pass)
#
# FIXME (mfh 2025/07/18) The test below fails to compile,
# and also SHOULD fail to compile.
# It looks like the test has some mark-up that turns it
# into a verify-that-it-fails-to-compile-as-expected test,
# but mdspan isn't finding that.
#
#mdspan_add_test(mdspan.conversion.verify)
mdspan_add_test(mdspan.ctor.copy.pass)
mdspan_add_test(mdspan.ctor.default.pass)
mdspan_add_test(mdspan.ctor.dh_array.pass)
mdspan_add_test(mdspan.ctor.dh_extents.pass)
mdspan_add_test(mdspan.ctor.dh_integers.pass)
mdspan_add_test(mdspan.ctor.dh_map.pass)
mdspan_add_test(mdspan.ctor.dh_map_acc.pass)
mdspan_add_test(mdspan.ctor.dh_span.pass)
mdspan_add_test(mdspan.ctor.move.pass)
mdspan_add_test(mdspan.deduction.pass)
#
# FIXME (mfh 2025/07/18) The test below fails to compile,
# and also SHOULD fail to compile.
# It looks like the test has some mark-up that turns it
# into a verify-that-it-fails-to-compile-as-expected test,
# but mdspan isn't finding that.
#
#mdspan_add_test(mdspan.element_type.verify)
#
# FIXME (mfh 2025/07/18) The test below fails to compile,
# and also SHOULD fail to compile.
# It looks like the test has some mark-up that turns it
# into a verify-that-it-fails-to-compile-as-expected test,
# but mdspan isn't finding that.
#
#mdspan_add_test(mdspan.extents.verify)
mdspan_add_test(mdspan.index_operator.pass)
#
# FIXME (mfh 2025/07/18) The test below fails to compile,
# and also SHOULD fail to compile.
# It looks like the test has some mark-up that turns it
# into a verify-that-it-fails-to-compile-as-expected test,
# but mdspan isn't finding that.
#
#mdspan_add_test(mdspan.mapping.verify)
mdspan_add_test(mdspan.move.pass)
mdspan_add_test(mdspan.properties.pass)
mdspan_add_test(mdspan.swap.pass)
mdspan_add_test(mdspan.types.pass)
