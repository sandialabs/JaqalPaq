# This example code demonstrates the behavior of run_jaqal_batch, run this file
# directly to see some example output which hopefully clarifies the behavior.
# Otherwise long comments with probably way too many redundancies are provided
# in the code itself.
from jaqalpaq.emulator import run_jaqal_string
import dataclasses


def get_max_list_length(overrides):
    """parameters lists must all be the same length in the end, so this is just
    a helper function to find the maximum length for verification and override
     expansion used in build_override_lists"""
    return max(
        map(lambda x: 1 if not isinstance(x, list) else len(x), overrides.values())
    )


def filter_system_parameters(overrides):
    """We want to ignore special parameters or pulse definition parameters which
    can be passed in but aren't used for overriding values of let parameters in
    the jaqal code itself. These parameters either start with 'pd.' for pulse
    definition (calibrated parameter) overrides or use the dunder convention
    for variables such as __index__ (for subcircuit indices) or __repeats__
    for number of experimental averages to run"""
    filtered_overrides = {}
    for k, v in overrides.items():
        if k.startswith("pd.") or (k.startswith("__") and k.endswith("__")):
            pass
        else:
            filtered_overrides[k] = v
    return filtered_overrides


def build_override_lists(overrides: dict):
    """Overrides come in the form of scalars and lists. If multiple lists are
    used they must either be length 1 or have matching length since the
    parameter lists are zipped together when running. When scalars are passed
    (or lists of length 1) they are repeated N times when other parameters are
    specified as lists with length == N.  This function essentailly expands all
    parameters into lists of identical length for simplified processing.

    If subcircuit indices are specified via '__index__' then there is some
    special handling for that in case each set of parameters needs to be run
    against multiple subcircuits.  Passing in a nested list of __index__ values
    is basically equivalent to the following (fairly terrible) pseudopython

    for parameter_set in zip(overrides['p1'], overrides['p2'], ...):
        jaqal_with_overrides = jaqal_code.replace_parameters(parameter_set)
        for index in overrides['__index__'][0]:
            yield jaqal_subcircuit_by_index(jaqal_with_overrides, index)

    thus some special handling is needed if nested lists are used for __index__.
    Currently we only support nested lists with an outer length of 1, though we
    could probably work in support for a nested list of length N.
    """
    maxlen = get_max_list_length(overrides)
    expanded_overrides = {}
    for k, v in overrides.items():
        if isinstance(v, list):
            if len(v) == 1:
                expanded_overrides[k] = v * maxlen
            elif len(v) != maxlen:
                raise Exception("Can't have mixed length lists in overrides")
            else:
                if k == "__index__" and isinstance(v[0], list):
                    raise Exception(
                        "Nested lists for '__index__' must have an outer list of length one: [[0,1,2,...]]"
                    )
                expanded_overrides[k] = v
        else:  # we have a scalar parameter which needs to be repeated
            expanded_overrides[k] = [v] * maxlen
    return expanded_overrides, maxlen


@dataclasses.dataclass
class DummyResult:
    """The results object always has a length one list,
    so this is just a dummy class to mimic that behavior
    """

    subcircuits: list = dataclasses.field(default_factory=list)


def ref_jaqal_batch(natural_count, expanded_overrides, override_count):
    filtered_overrides = filter_system_parameters(expanded_overrides)
    by_batch = []
    as_executed = []
    for idx in range(override_count):
        local_OR_dict = {k: v[idx] for k, v in filtered_overrides.items()}
        batch = []
        by_batch.append((local_OR_dict, batch))
        if "__index__" in expanded_overrides:
            if isinstance(expanded_overrides["__index__"][idx], list):
                for subcircuit_index in expanded_overrides["__index__"][idx]:
                    as_executed.append((local_OR_dict, subcircuit_index))
                    batch.append(subcircuit_index)
            else:
                batch.append(expanded_overrides["__index__"][idx])
                as_executed.append(
                    (local_OR_dict, expanded_overrides["__index__"][idx])
                )
        else:
            for i in range(natural_count):
                batch.append(i)
                as_executed.append((local_OR_dict, i))

    return by_batch, as_executed


def run_jaqal_batch(
    jaqal_string: str, overrides: dict
):  # -> a generator of jaqalpaq emulation results
    """A dummy version of run_jaqal_batch. This iterates over parameters passed
    in via the 'overrides' input dictionary. Conventionally, we handle inputs
    which can start with 'j.' or not, but I think we can forego this level of
    handling since that is tied with namespace parameters used in IonControl and
    has no real meaning here. We do support passing of pulse definition
    parameters using either 'j.pd.xxx' or 'pd.xxx' naming patterns, but those
    could also be ignored
    """
    # expand all overrides into lists of a common length
    expanded_overrides, override_count = build_override_lists(overrides)
    # filter out system parameters which have dunder names. In this example I'm
    # not handling "__repeats__", which specifies the number of experimental
    # averages to take
    filtered_overrides = filter_system_parameters(expanded_overrides)
    for idx in range(override_count):
        # create a local override dict where parameter lists are effectively
        # zipped together and scanned over a common index
        local_OR_dict = {k: v[idx] for k, v in filtered_overrides.items()}
        result = run_jaqal_string(jaqal_string.format(**local_OR_dict))
        if "__index__" in expanded_overrides:
            # if __index__ is specified, then we need to handle a few cases:
            #   1) scalar: only one subcircuit is run for the corresponding overrides
            #   2) flat list: subcircuits are zipped with override lists
            #   3) nested list: the subcircuits in the inner list are iterated
            #                   over for each element of the zipped parameter lists
            #           NOTE: currently nested lists are only supported for an
            #                 outer list of length 1 (ie [[0,1,2...]]) but in
            #                 this example the lists are expanded under the hood.
            #                 This behavior could probably change if needed
            if isinstance(expanded_overrides["__index__"][idx], list):
                # If we have a nested list, we need to repeat the result for
                # multiple overrides we basically have to unpack the subcircuits
                # manually since someone might request repeated indices of a
                # subcircuit (e.g. [[0,1,0]]) so we can't use the same
                # subcircuit[idx] convention
                for subcircuit_index in expanded_overrides["__index__"][idx]:
                    print(
                        f"yielding subcircuit {subcircuit_index}, Overrides: {local_OR_dict}, Results:",
                        end=" ",
                    )
                    yield DummyResult([result.subcircuits[subcircuit_index]])
            else:
                # We need to yield an individual subcircuit for which the index was specified
                print(
                    f"yielding subcircuit {expanded_overrides['__index__'][idx]}, Overrides: {local_OR_dict}, Results:",
                    end=" ",
                )
                yield DummyResult(
                    [result.subcircuits[expanded_overrides["__index__"][idx]]]
                )
        else:
            # The default behavior is to run each parameter for all subcircuits,
            # however these need to be unrolled in length 1 result.subcircuits[]
            # lists to maintain consistency with passing a nested list
            for i, subcirc in enumerate(result.subcircuits):
                print(
                    f"yielding subcircuit {i}, Overrides: {local_OR_dict}, Results:",
                    end=" ",
                )
                yield DummyResult([subcirc])
