# Loss functions

This package provides standard and method-specific loss modules. Importing
[__init__.py](__init__.py) registers them in [LOSSFUNC](../tools/registry.py).
Detectors select the inputs, combine loss terms, and return the optimization loss
as `losses['overall']`.

| Module | Contents |
|---|---|
| [abstract_loss.py](abstract_loss.py) | Shared loss interface |
| [standard_losses](standard_losses) | Cross-entropy, MSE, and Wasserstein objectives |
| [customized_losses](customized_losses) | AVFF, FRADE, and MCL objectives |

## Configuration

```yaml
model_config:
  losses:
    - name: CE
      func: cross_entropy
      weight: 1.0
      params: {}
```

`func` selects the registry key and `params` supplies constructor arguments.
`name` is the detector's lookup name, so use the names expected by that detector.
The detector controls how the outer `weight` is applied; there is no global loss
builder that applies it automatically. Forward signatures and reductions are
documented in each implementation.

## Adding a loss

Implement an `nn.Module`, usually derived from `AbstractLoss`, decorate it with
`@LOSSFUNC.register_module(module_name='your_loss')`, and import it in this
package's initializer. Add the configuration entry and invoke it from the
relevant detector. Handle structured loss outputs explicitly.
