//! Custom [`rootcause`] handlers used to replace functionality that the
//! `error-stack` API exposed directly.
//!
//! These handlers are wired into [`rootcause::Report`] via
//! [`new_custom`](rootcause::Report::new_custom) and
//! [`attach_custom`](rootcause::Report::attach_custom).

use core::fmt;

use rootcause::handlers::{AttachmentFormattingStyle, AttachmentHandler, FormattingFunction};

/// Location attachment and handler shipped with `rootcause`, used by
/// [`IntoContext`](crate::IntoContext).
pub(crate) use rootcause::hooks::builtin_hooks::location::{Location, LocationHandler};

/// Attachment handler that always formats its attachment using [`Debug`].
///
/// This is a handler-based equivalent of wrapping a value in a
/// [`Debug`]-formatting newtype.
///
/// [`Debug`]: core::fmt::Debug
#[derive(Debug, Copy, Clone)]
pub struct DebugFmt;

impl<A> AttachmentHandler<A> for DebugFmt
where
    A: fmt::Debug,
{
    fn display(value: &A, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(value, f)
    }

    fn debug(value: &A, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(value, f)
    }

    fn preferred_formatting_style(
        _value: &A,
        _report_formatting_function: FormattingFunction,
    ) -> AttachmentFormattingStyle {
        AttachmentFormattingStyle {
            function: FormattingFunction::Debug,
            ..Default::default()
        }
    }
}
